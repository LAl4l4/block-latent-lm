"""Two-block lookahead with exact replay of emitted text feedback."""

import torch

from latent_ar.feedback_student import FeedbackStudent


@torch.inference_mode()
def generate_pair_feedback(
    teacher,
    student,
    prompt,
    lengths,
    max_new_tokens=64,
    generator=None,
    min_new_tokens=0,
    return_trace=False,
    parallel=False,
    verify_repair=False,
):
    if not isinstance(student, FeedbackStudent):
        raise ValueError("Pair feedback requires an MLP FeedbackStudent")
    if max_new_tokens < 1 or not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("Invalid generation lengths")
    if parallel not in (False, "batch"):
        raise ValueError(
            "parallel must be False or 'batch'; CUDA streams are not implemented"
        )
    if student.cfg.noise_dim not in (0, teacher.cfg.chunk_size):
        raise ValueError("Student noise dimension must be zero or chunk_size")
    if student.cfg.noise_dim == 0 and teacher.cfg.token_temperature != 0:
        raise ValueError("Deterministic student requires greedy token sampling")
    if (
        student.cfg.context_dim != teacher.cfg.dim
        or student.cfg.feedback_dim != teacher.cfg.dim
    ):
        raise ValueError("Student dimensions must match teacher backbone")
    teacher_was, student_was = teacher.training, student.training
    teacher.eval()
    student.eval()
    try:
        rng, _ = teacher.generation_generators(generator, prompt.device)
        encoded, valid = teacher.encode_history(prompt, lengths)
        batch, width, _ = encoded.shape
        n_history = valid.sum(-1)
        positions = torch.arange(width, device=prompt.device).expand(batch, -1)
        hidden, cache = teacher.backbone(encoded, positions, valid)
        context = hidden[torch.arange(batch, device=prompt.device), n_history - 1]
        all_inputs, all_positions = [encoded], [positions]
        finished = torch.zeros(batch, dtype=torch.bool, device=prompt.device)
        parts, records = [], []
        chunk = teacher.cfg.chunk_size
        for start in range(0, max_new_tokens, 2 * chunk):
            count = min(2, (max_new_tokens - start + chunk - 1) // chunk)
            noises = [
                teacher.draw_noise((batch,), prompt.device, rng) for _ in range(count)
            ]
            predicted = (
                student(context, noises[0] if student.cfg.noise_dim else None)
                if count == 2
                else None
            )
            contexts = [context]
            if count == 2:
                predicted_valid = torch.cat((valid, (~finished)[:, None]), 1)
                provisional, _ = teacher.backbone(
                    predicted[:, None],
                    (n_history + start // chunk)[:, None],
                    predicted_valid,
                    cache,
                )
                contexts.append(provisional[:, 0])
            if parallel == "batch" and count == 2:
                z = teacher.sample_latent(torch.cat(contexts), noise=torch.cat(noises))
                logits = list(teacher.codec.decode(z[:, None]).split(batch))
                zs = list(z.split(batch))
            else:
                zs = [
                    teacher.sample_latent(h, noise=u) for h, u in zip(contexts, noises)
                ]
                logits = [teacher.codec.decode(z[:, None]) for z in zs]
            feedbacks, masks, group_records = [], [], []
            for index in range(count):
                offset = start + index * chunk
                active = ~finished
                block_width = min(chunk, max_new_tokens - offset)
                scores = logits[index][:, :block_width].float().clone()
                scores[..., teacher.cfg.pad_id] = -torch.inf
                scores[..., teacher.cfg.bos_id] = -torch.inf
                slots = torch.arange(offset, offset + block_width, device=prompt.device)
                scores[:, slots < min_new_tokens, teacher.cfg.eos_id] = -torch.inf
                chosen = scores.argmax(-1)
                actual_lengths = torch.zeros(
                    batch, dtype=torch.long, device=prompt.device
                )
                emitted = []
                for slot in range(block_width):
                    alive = ~finished
                    emitted.append(
                        torch.where(alive, chosen[:, slot], teacher.cfg.eos_id)
                    )
                    actual_lengths += alive.long()
                    finished |= alive & (chosen[:, slot] == teacher.cfg.eos_id)
                    if bool(finished.all()):
                        break
                tokens = torch.stack(emitted, 1)
                parts.append(tokens)
                ids = tokens.masked_fill(
                    torch.arange(tokens.shape[1], device=prompt.device)[None]
                    >= actual_lengths[:, None],
                    teacher.cfg.pad_id,
                )
                feedback, _ = teacher.encode_history(ids, actual_lengths.clamp_min(1))
                feedback = feedback[:, 0].masked_fill(~active[:, None], 0)
                feedbacks.append(feedback)
                masks.append(~finished)
                if return_trace:
                    group_records.append(
                        dict(
                            context=contexts[index].clone(),
                            noise=noises[index].clone(),
                            z=zs[index].clone(),
                            token_ids=tokens.clone(),
                            lengths=actual_lengths.clone(),
                            active=active.clone(),
                            feedback=feedback.clone(),
                            predicted_feedback=(
                                predicted.clone()
                                if index == 0 and predicted is not None
                                else None
                            ),
                        )
                    )
                if bool(finished.all()):
                    break
            if return_trace:
                records.extend(group_records)
            if bool(finished.all()) or start + count * chunk >= max_new_tokens:
                break
            inputs = torch.stack(feedbacks, 1)
            new_valid = torch.stack(masks, 1)
            new_positions = (
                n_history[:, None]
                + torch.arange(
                    start // chunk,
                    start // chunk + len(feedbacks),
                    device=prompt.device,
                )[None]
            )
            valid = torch.cat((valid, new_valid), 1)
            hidden, cache = teacher.backbone(inputs, new_positions, valid, cache)
            context = hidden[:, -1]
            if verify_repair:
                all_inputs.append(inputs)
                all_positions.append(new_positions)
                full_h, full_cache = teacher.backbone(
                    torch.cat(all_inputs, 1), torch.cat(all_positions, 1), valid
                )
                # Finished rows cannot contribute to a future context, and invalid
                # key slots are masked from every subsequent attention operation.
                errors = [
                    (context.float() - full_h[:, -1].float())
                    .abs()
                    .masked_fill(finished[:, None], 0)
                    .max()
                ]
                errors.extend(
                    (a.float() - b.float())
                    .abs()
                    .masked_fill(~valid[:, None, :, None], 0)
                    .max()
                    for c, d in zip(cache, full_cache)
                    for a, b in zip(c, d)
                )
                error = torch.stack(errors).max().item()
                if return_trace:
                    records[-1]["max_cache_error"] = error
            if return_trace:
                records[-1]["repair_inputs"] = inputs.clone()
                records[-1]["repair_valid"] = new_valid.clone()
                records[-1]["repair_positions"] = new_positions.clone()
                records[-1]["repair_context"] = context.clone()
        output = torch.cat(parts, 1)
        return {"output_ids": output, "records": records} if return_trace else output
    finally:
        teacher.train(teacher_was)
        student.train(student_was)
