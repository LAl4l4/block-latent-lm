"""逐块解码并对照文字回灌与原始 latent 回灌。"""

import torch


@torch.inference_mode()
def generate_with_feedback(
    model,
    prompt,
    prompt_lengths,
    max_new_tokens=64,
    temperature=0.0,
    top_k=50,
    use_latents=True,
    generator=None,
    min_new_tokens=0,
    feedback_mode="text",
    return_trace=False,
    record_sampling_noise=False,
):
    """逐块预测、确定性解码并回灌；raw 模式也逐块解码，不能作为异步速度基线。"""
    if not use_latents:
        raise ValueError("Feedback generation requires predicted latents")
    if max_new_tokens < 1 or not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("Invalid generation lengths")
    if temperature != 0 or top_k < 0:
        raise ValueError("Feedback generation requires temperature=0 and top_k>=0")
    if feedback_mode not in {"text", "raw"}:
        raise ValueError("feedback_mode must be 'text' or 'raw'")

    was_training = model.training
    model.eval()
    try:
        latent_rng, _ = model.generation_generators(generator, prompt.device)
        # CALM 使用可训练的历史压缩器，其他模型继续使用原 codec encoder。
        encode_history = getattr(model, "encode_history", model.codec.encode)
        encoded, valid = encode_history(prompt, prompt_lengths)
        batch, width, _ = encoded.shape
        lengths = valid.sum(-1)
        positions = torch.arange(width, device=prompt.device).expand(batch, -1)
        hidden, caches = model.backbone(encoded, positions, valid)
        context = hidden[torch.arange(batch, device=prompt.device), lengths - 1]
        finished = torch.zeros(batch, dtype=torch.bool, device=prompt.device)
        output_parts = []
        records = [] if return_trace else None

        for start in range(0, max_new_tokens, model.cfg.chunk_size):
            active = ~finished
            if record_sampling_noise:
                noise = model.draw_noise((batch,), prompt.device, latent_rng)
                z = model.sample_latent(context, noise=noise)
            else:
                z = model.sample_latent(context, generator=latent_rng)
            width = min(model.cfg.chunk_size, max_new_tokens - start)
            scores = model.codec.decode(z[:, None])[:, :width].float().clone()
            scores[..., model.cfg.pad_id] = -torch.inf
            scores[..., model.cfg.bos_id] = -torch.inf
            indices = torch.arange(start, start + width, device=prompt.device)
            scores[:, indices < min_new_tokens, model.cfg.eos_id] = -torch.inf
            chosen = scores.argmax(-1)
            actual_lengths = torch.zeros(batch, dtype=torch.long, device=prompt.device)
            emitted = []
            for offset in range(width):
                alive = ~finished
                token = torch.where(alive, chosen[:, offset], model.cfg.eos_id)
                actual_lengths += alive.long()
                emitted.append(token)
                finished |= alive & (token == model.cfg.eos_id)
                if bool(finished.all()):
                    break
            tokens = torch.stack(emitted, dim=1)
            output_parts.append(tokens)

            # raw 模式直接使用预测向量；text 模式随后采用历史编码的类型与宽度。
            feedback = z.clone()
            if feedback_mode == "text":
                # codec 要求长度为正，只编码本轮开始时仍活跃的行。
                rows = active.nonzero(as_tuple=True)[0]
                ids = tokens[rows].clone()
                mask = (
                    torch.arange(ids.shape[1], device=ids.device)[None]
                    >= actual_lengths[rows, None]
                )
                ids.masked_fill_(mask, model.cfg.pad_id)
                encoded_text, _ = encode_history(ids, actual_lengths[rows])
                # autocast 下生成头可能输出 BF16，而历史压缩器输出 FP32。
                # 按历史编码分配，既保留其精度，也允许历史维度与生成 latent 不同。
                # 已结束行用零占位，后续缓存掩码会排除这些行。
                feedback = encoded_text.new_zeros((batch, encoded_text.shape[-1]))
                feedback[rows] = encoded_text[:, 0]
            if return_trace:
                records.append(
                    {
                        "z": z.detach().clone(),
                        "feedback": feedback.detach().clone(),
                        "token_ids": tokens.detach().clone(),
                        "lengths": actual_lengths.detach().clone(),
                        "active": active.detach().clone(),
                        "context": context.detach().clone(),
                    }
                )
                if record_sampling_noise:
                    records[-1]["noise"] = noise.detach().clone()
            if bool(finished.all()) or start + width >= max_new_tokens:
                break
            # 已结束行仍占 cache 槽位，但不能作为有效的注意力 key。
            continuing = ~finished
            valid = torch.cat([valid, continuing[:, None]], dim=1)
            hidden, caches = model.backbone(
                feedback[:, None],
                (lengths + start // model.cfg.chunk_size)[:, None],
                valid,
                caches,
            )
            context = hidden[:, 0]

        output = torch.cat(output_parts, dim=1)
        return {"output_ids": output, "records": records} if return_trace else output
    finally:
        model.train(was_training)
