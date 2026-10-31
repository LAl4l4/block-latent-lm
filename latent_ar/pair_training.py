"""两块一组前瞻的训练前向：训练条件与 pair_feedback 推理调度一致。

生成块按 (0,1), (2,3), ... 分组。组首块（偶数块）读取全部真实反馈；
组内第二块（奇数块 t）的主干输入里，第 t-1 块位置不是真实文字反馈
E(x_{t-1})，而是 q_{t-1} = Q(h_{t-1})。h_{t-1} 本身只依赖第 t-2 块及以前
的真实文字，所以奇数块的条件状态不读取第 t-1 块的实际文字。

mode:
  sync  - 原教师：每块都读取上一块真实反馈（基线）。
  state - Q 是读取 h_{t-1} 的 FeedbackStudent，可经语言模型损失端到端训练。
  const - Q 只输出一个可学习常量（输出层权重固定为 0，只训练 bias）。
"""

import torch
import torch.nn.functional as F

from latent_ar.calm import shuffle_valid_contexts

MODES = ("sync", "state", "const")


def history_layout(model, prompt, prompt_lengths, labels):
    """与 CalmModel.teacher_contexts 相同的主干输入布局，额外返回中间量。"""
    cfg = model.cfg
    lengths = labels.ne(-100).sum(-1)
    if bool((lengths < 1).any()):
        raise ValueError("Every target must contain at least one token")
    ids = labels.masked_fill(labels.eq(-100), cfg.pad_id)
    with torch.no_grad():
        mean, _, target_valid = model.codec.encode_posterior(ids, lengths)
    prompt_encoded, prompt_valid = model.encode_history(prompt, prompt_lengths)
    target_encoded, _ = model.encode_history(ids, lengths)
    batch, prompt_width, _ = prompt_encoded.shape
    steps = mean.shape[1]
    prompt_blocks = prompt_valid.sum(-1)
    history = torch.cat((prompt_encoded, target_encoded[:, :-1]), 1)
    valid = torch.cat((prompt_valid, target_valid[:, :-1]), 1)
    positions = torch.cat(
        (
            torch.arange(prompt_width, device=prompt.device).expand(batch, -1),
            prompt_blocks[:, None] + torch.arange(steps - 1, device=prompt.device),
        ),
        1,
    )
    indices = torch.cat(
        (
            (prompt_blocks - 1)[:, None],
            torch.arange(
                prompt_width, prompt_width + steps - 1, device=prompt.device
            ).expand(batch, -1),
        ),
        1,
    )
    return {
        "history": history,
        "valid": valid,
        "positions": positions,
        "indices": indices,
        "prompt_width": prompt_width,
        "steps": steps,
        "target_valid": target_valid,
        "target_feedback": target_encoded,
    }


def gather_contexts(hidden, indices):
    return hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))


def pair_contexts(model, student, layout, mode):
    """返回每个目标块实际使用的条件状态 [batch, steps, dim]。"""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    hidden, _ = model.backbone(layout["history"], layout["positions"], layout["valid"])
    contexts = gather_contexts(hidden, layout["indices"])
    steps, width = layout["steps"], layout["prompt_width"]
    odd = list(range(1, steps, 2))
    if mode == "sync" or not odd:
        return contexts, {}
    batch = contexts.shape[0]
    rows, predictions = [], {}
    for t in odd:
        slot = width + t - 1
        q = student(contexts[:, t - 1]).to(layout["history"].dtype)
        predictions[t - 1] = q
        history = layout["history"]
        # 只替换第 t-1 块的位置；因果 mask 保证更后的位置不影响第 slot 位输出。
        rows.append(torch.cat((history[:, :slot], q[:, None], history[:, slot + 1 :]), 1))
    hidden2, _ = model.backbone(
        torch.cat(rows, 0),
        layout["positions"].repeat(len(odd), 1),
        layout["valid"].repeat(len(odd), 1),
    )
    pieces = hidden2.split(batch)
    columns = [contexts[:, t] for t in range(steps)]
    for piece, t in zip(pieces, odd):
        columns[t] = piece[:, width + t - 1]
    return torch.stack(columns, 1), predictions


def block_targets(model, labels, steps):
    cfg = model.cfg
    blocks = F.pad(
        labels, (0, steps * cfg.chunk_size - labels.shape[1]), value=-100
    ).reshape(labels.shape[0], steps, cfg.chunk_size)
    previous = torch.full_like(blocks, cfg.pad_id)
    previous[:, :, 0] = cfg.bos_id
    previous[:, :, 1:] = blocks[:, :, :-1].masked_fill(
        blocks[:, :, :-1].eq(-100), cfg.pad_id
    )
    return blocks, previous


def pair_statistics(
    model, student, prompt, prompt_lengths, labels, mode, shuffle_context=False
):
    """块级 CE 求和；按组首块（偶数）与组内第二块（奇数）分别统计。

    shuffle_context 与教师 context_advantage 同口径：同一块序号的条件状态在
    batch 内错配，用来给出"整段历史都错"时每类块的 CE 上界。
    """
    layout = history_layout(model, prompt, prompt_lengths, labels)
    contexts, predictions = pair_contexts(model, student, layout, mode)
    if shuffle_context:
        contexts = shuffle_valid_contexts(contexts, layout["target_valid"])
    blocks, previous = block_targets(model, labels, layout["steps"])
    logits = model.latent_head(contexts, previous)
    token_nll = F.cross_entropy(
        logits.float().flatten(0, 2),
        blocks.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view(blocks.shape)
    present = blocks.ne(-100)
    odd = (torch.arange(layout["steps"], device=labels.device) % 2).bool()
    odd_blocks = odd[None, :, None].expand_as(present)
    first = torch.zeros_like(present)
    first[:, :, 0] = True
    out = {
        "objective_sum": token_nll.sum(),
        "tokens": int(present.sum()),
        "blocks": int(layout["target_valid"].sum()),
        "even_nll": float(token_nll[~odd_blocks].sum()),
        "even_tokens": int((present & ~odd_blocks).sum()),
        "odd_nll": float(token_nll[odd_blocks].sum()),
        "odd_tokens": int((present & odd_blocks).sum()),
        "odd_first_nll": float(token_nll[odd_blocks & first].sum()),
        "odd_first_tokens": int((present & odd_blocks & first).sum()),
        "even_first_nll": float(token_nll[~odd_blocks & first].sum()),
        "even_first_tokens": int((present & ~odd_blocks & first).sum()),
    }
    if predictions:
        # 诊断：Q 输出与真实文字反馈的距离（只看被替换且有效的块）。
        errors, norms, count = 0.0, 0.0, 0
        for index, q in predictions.items():
            keep = layout["target_valid"][:, index + 1]
            if bool(keep.any()):
                target = layout["target_feedback"][:, index][keep].float()
                diff = q[keep].float() - target
                errors += float(diff.pow(2).sum(-1).sqrt().sum())
                norms += float(target.pow(2).sum(-1).sqrt().sum())
                count += int(keep.sum())
        out["q_rel_error_sum"], out["q_norm_sum"], out["q_count"] = errors, norms, count
    return out


def make_constant(student):
    """把 FeedbackStudent 变成常量输出：输出层权重置零并冻结，只训练 bias。"""
    with torch.no_grad():
        student.output.weight.zero_()
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name == "output.bias")
    return student
