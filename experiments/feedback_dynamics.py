"""Fixed-history feedback behavior and differentiable teacher distillation."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class FeedbackHistory:
    caches: list
    valid: torch.Tensor
    next_position: torch.Tensor


def prepare_feedback_history(teacher, prompt_ids, prompt_lengths, past_feedback, past_lengths):
    """Prefill only prompt and already-fed feedback; lengths describe each row."""
    teacher.eval().requires_grad_(False)
    if (prompt_lengths < 1).any() or (past_lengths < 0).any() or (past_lengths > past_feedback.shape[1]).any():
        raise ValueError('Invalid history lengths')
    with torch.no_grad():
        prompt, prompt_valid = teacher.encode_history(prompt_ids, prompt_lengths)
        counts = prompt_valid.sum(-1)
        lengths = counts + past_lengths
        width = int(lengths.max())
        packed = prompt.new_zeros((len(prompt), width, prompt.shape[-1]))
        for i in range(len(prompt)):
            n, p = int(counts[i]), int(past_lengths[i])
            packed[i, :n] = prompt[i, :n]
            packed[i, n:n+p] = past_feedback[i, :p]
        positions = torch.arange(width, device=prompt.device).expand(len(prompt), -1)
        valid = positions < lengths[:, None]
        _, caches = teacher.backbone(packed, positions, valid)
    return FeedbackHistory(caches, valid, lengths)


def next_block_logits(teacher, history, feedback, next_tokens):
    valid = torch.cat((history.valid, torch.ones((len(feedback), 1), dtype=torch.bool, device=feedback.device)), 1)
    hidden, _ = teacher.backbone(feedback[:, None], history.next_position[:, None], valid, history.caches)
    previous = torch.full_like(next_tokens, teacher.cfg.pad_id)
    previous[:, 0] = teacher.cfg.bos_id
    previous[:, 1:] = next_tokens[:, :-1].masked_fill(next_tokens[:, :-1].eq(-100), teacher.cfg.pad_id)
    # cuDNN does not support backward through an eval-mode GRU. Keep the
    # frozen teacher in eval mode while using the differentiable native path.
    with torch.backends.cudnn.flags(enabled=False):
        return teacher.latent_head(hidden[:, 0], previous)


def behavior_distillation_loss(teacher, history, prediction, target, next_tokens, mse_weight=0.1):
    if mse_weight < 0:
        raise ValueError('mse_weight must be nonnegative')
    mask = next_tokens.ne(-100)
    if not bool(mask.any()):
        raise ValueError('No valid next tokens')
    with torch.no_grad():
        reference = next_block_logits(teacher, history, target, next_tokens).float()
    logits = next_block_logits(teacher, history, prediction, next_tokens).float()
    logp = F.log_softmax(reference, -1)
    kl = (logp.exp() * (logp - F.log_softmax(logits, -1))).sum(-1)[mask].mean()
    mse = F.mse_loss(prediction.float(), target.float())
    return kl + mse_weight * mse, {'kl': kl.detach(), 'mse': mse.detach(), 'valid_tokens': mask.sum().detach()}


def main():
    import argparse
    import json
    from pathlib import Path
    from torch.nn.utils.rnn import pad_sequence
    from experiments.distill_gru_feedback import load_teacher
    from latent_ar.feedback_student import load_feedback_student, AttentionFeedbackStudent, materialize_history_windows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--students', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--blocks', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2033)
    parser.add_argument('--random-repeats', type=int, default=3)
    args = parser.parse_args()
    if args.blocks < 1 or args.random_repeats < 1 or args.batch_size < 1:
        parser.error('blocks, batch-size and random-repeats must be positive')
    device = torch.device(args.device)
    from download_data import sha256
    from train import precision_context
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output}')
    teacher, manifest = load_teacher(args.data, device)
    snapshot = torch.load(args.data / 'teacher.pt', map_location='cpu', weights_only=True, mmap=True)
    precision = snapshot['config']['training']['precision']
    autocast = precision_context(device, precision)
    del snapshot
    if sha256(args.data / (args.split + '.pt')) != manifest['split_sha256'][args.split]:
        raise ValueError('Diagnostic split hash mismatch')
    if sha256(args.data / 'train.pt') != manifest['split_sha256']['train']:
        raise ValueError('Training split hash mismatch')
    teacher.requires_grad_(False)
    data = torch.load(args.data / (args.split + '.pt'), map_location='cpu', weights_only=True)
    n = min(args.blocks, len(data['h']))
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(data['h']), generator=rng)[:n]
    # Mean feedback is estimated from training data, never the selected test rows.
    train = torch.load(args.data / 'train.pt', map_location='cpu', weights_only=True)
    mean = train['r'].float().mean(0).to(device)
    students = []
    for path in args.students:
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        source_hash = saved.get('study_settings', {}).get('teacher_hash', saved.get('teacher_snapshot_sha256'))
        if source_hash != manifest['teacher_snapshot_sha256']:
            raise ValueError(f'{path}: student teacher hash mismatch or missing')
        student = load_feedback_student(saved).to(device).eval()
        settings = saved.get('study_settings', {})
        mode = settings.get('noise_mode', 'shared' if student.cfg.noise_dim else 'none')
        windows = materialize_history_windows(data, student.cfg.history_blocks, student.cfg.use_prompt_history) if isinstance(student, AttentionFeedbackStudent) else None
        students.append((str(path), student, mode, windows))
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'samples.jsonl'
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with output.open('w', encoding='utf-8') as stream, torch.no_grad(), autocast():
        for start in range(0, n, args.batch_size):
            idx = indices[start:start + args.batch_size]
            prompts = [data['prompts'][int(i)] for i in data['prompt_row'][idx]]
            lengths = torch.tensor([len(p) for p in prompts], device=device)
            ids = pad_sequence(prompts, batch_first=True, padding_value=teacher.cfg.pad_id).to(device)
            history = prepare_feedback_history(teacher, ids, lengths, data['past_feedback'][idx].to(device), data['past_lengths'][idx].to(device))
            h, r, u, tokens = (data[k][idx].to(device) for k in ('h', 'r', 'u', 'next_tokens'))
            ref = next_block_logits(teacher, history, r, tokens).float()
            logp = F.log_softmax(ref, -1)
            mask = tokens.ne(-100)
            count = mask.sum(-1).clamp_min(1)
            ref_ce = F.cross_entropy(ref.transpose(1, 2), tokens, ignore_index=-100, reduction='none').sum(-1) / count
            candidates = [('true', r), ('mean', mean.expand_as(r))]
            for name, student, mode, windows in students:
                if windows is not None:
                    pred = student(windows[0][idx].to(device), valid=windows[1][idx].to(device))
                else:
                    noise = None if student.cfg.noise_dim == 0 else (u if mode == 'shared' else torch.full_like(u, .5))
                    pred = student(h, noise)
                candidates.append((name, pred))
                error = pred - r
                norm = error.norm(dim=-1, keepdim=True)
                direction = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12) * norm
                randoms = []
                for rep in range(args.random_repeats):
                    v = torch.randn(r.shape, generator=generator, device=device, dtype=r.dtype)
                    randoms.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * norm)
                for scale in (0., .25, .5, 1., 2.):
                    candidates.append((f'{name}:error:{scale}', r + scale * error))
                    candidates.append((f'{name}:feedback_direction:{scale}', r + scale * direction))
                    candidates.extend((f'{name}:random{rep}:{scale}', r + scale * v) for rep, v in enumerate(randoms))
            for label, feedback in candidates:
                logits = next_block_logits(teacher, history, feedback, tokens).float()
                kl = (logp.exp() * (logp - F.log_softmax(logits, -1))).sum(-1).masked_fill(~mask, 0).sum(-1) / count
                ce = F.cross_entropy(logits.transpose(1, 2), tokens, ignore_index=-100, reduction='none').sum(-1) / count
                err = feedback - r
                changed = logits[:, 0].argmax(-1).ne(ref[:, 0].argmax(-1))
                for j, i in enumerate(idx.tolist()):
                    record = {'sample_index': i, 'story_id': int(data['story_id'][i]), 'block_index': int(data['block_index'][i]), 'condition': label, 'mse': float(err[j].square().mean()), 'error_norm': float(err[j].norm()), 'feedback_norm': float(feedback[j].norm()), 'target_norm': float(r[j].norm()), 'kl': float(kl[j]), 'reference_token_ce': float(ce[j]), 'reference_token_ce_delta': float(ce[j] - ref_ce[j]), 'first_argmax_changed': bool(changed[j]), 'valid_tokens': int(count[j])}
                    stream.write(json.dumps(record) + '\n')
    (args.output / 'config.json').write_text(json.dumps({'data': str(args.data), 'students': [str(x) for x in args.students], 'split': args.split, 'blocks': n, 'seed': args.seed, 'random_repeats': args.random_repeats, 'teacher_hash': manifest['teacher_snapshot_sha256'], 'precision': precision, 'effective_precision': precision if device.type == 'cuda' else 'fp32', 'note': 'Fixed history; CE uses teacher-generated words, not ground truth.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

