"""Probe whether shared sampling noise predicts actual feedback at fixed contexts."""
import argparse
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer
from download_data import sha256
from experiments.distill_gru_feedback import load_teacher
from latent_ar.feedback_student import load_feedback_student
from train import precision_context


@torch.no_grad()
def draw_feedback(teacher, h, u):
    """Decode sampled content, truncate at EOS, and encode exactly emitted words."""
    z = teacher.sample_latent(h, noise=u)
    scores = teacher.codec.decode(z[:, None]).float().clone()
    scores[..., teacher.cfg.pad_id] = -torch.inf
    scores[..., teacher.cfg.bos_id] = -torch.inf
    words = scores.argmax(-1)
    eos = words.eq(teacher.cfg.eos_id)
    has_eos = eos.any(-1)
    first = eos.long().argmax(-1)
    lengths = torch.where(has_eos, first + 1, torch.full_like(first, words.shape[1]))
    valid = torch.arange(words.shape[1], device=words.device)[None] < lengths[:, None]
    emitted = words.masked_fill(~valid, teacher.cfg.pad_id)
    feedback, _ = teacher.encode_history(emitted, lengths)
    return emitted, lengths, feedback[:, 0], ~has_eos


def mean_stderr(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if not len(values):
        return {'mean': None, 'stderr': None, 'count': 0}
    return {'mean': float(values.mean()), 'stderr': float(values.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else None, 'count': len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--students', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contexts', type=int, default=32)
    parser.add_argument('--draws', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2041)
    args = parser.parse_args()
    if min(args.contexts, args.draws) < 1:
        parser.error('contexts and draws must be positive')
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    manifest = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in [('teacher.pt', manifest['teacher_snapshot_sha256']), ('test.pt', manifest['split_sha256']['test']), ('tokenizer.json', manifest['tokenizer_sha256'])]:
        if sha256(args.data / name) != expected:
            raise ValueError(f'{name} hash mismatch')
    teacher, _ = load_teacher(args.data, args.device)
    snapshot = torch.load(args.data / 'teacher.pt', map_location='cpu', weights_only=True, mmap=True)
    precision = snapshot['config']['training']['precision']
    autocast = precision_context(torch.device(args.device), precision)
    del snapshot
    teacher.requires_grad_(False)
    if teacher.cfg.token_temperature <= 0:
        raise ValueError('Probe requires a stochastic teacher')
    data = torch.load(args.data / 'test.pt', map_location='cpu', weights_only=True, mmap=True)
    indices = data['block_index'].eq(0).nonzero(as_tuple=True)[0][:args.contexts]
    if not len(indices):
        raise ValueError('No initial contexts')
    # Generate both paired and independent noise before observing any outcomes.
    rng = torch.Generator().manual_seed(args.seed)
    shape = (len(indices), args.draws, teacher.cfg.chunk_size)
    noise = torch.rand(shape, generator=rng)
    independent = torch.rand(shape, generator=rng)
    device = torch.device(args.device)
    students = []
    for path in args.students:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        source_hash = checkpoint.get('study_settings', {}).get('teacher_hash', checkpoint.get('teacher_snapshot_sha256'))
        if source_hash != manifest['teacher_snapshot_sha256']:
            raise ValueError(f'{path}: student teacher hash mismatch or missing')
        student = load_feedback_student(checkpoint).to(device).eval()
        mode = checkpoint.get('study_settings', {}).get('noise_mode')
        if mode not in ('shared', 'blind') or student.cfg.noise_dim != teacher.cfg.chunk_size:
            raise ValueError(f'{path}: expected shared/blind student with matching noise dimension')
        students.append((str(path), student, mode))
    tokenizer = Tokenizer.from_file(str(args.data / 'tokenizer.json'))
    records = []
    with torch.no_grad(), autocast():
        for row, index in enumerate(indices.tolist()):
            h = data['h'][index].to(device).expand(args.draws, -1)
            u = noise[row].to(device)
            words, lengths, feedback, continuing = draw_feedback(teacher, h, u)
            token_lists = [words[j, :lengths[j]].tolist() for j in range(args.draws)]
            record = {'sample_index': index, 'story_id': int(data['story_id'][index]), 'prompt_row': int(data['prompt_row'][index]), 'unique_contents': len({tuple(x) for x in token_lists}), 'continuing_count': int(continuing.sum()), 'eos_count': int((~continuing).sum()), 'feedback_variance_all': float(feedback.float().var(0, unbiased=False).mean()), 'feedback_variance_continuing': float(feedback[continuing].float().var(0, unbiased=False).mean()) if continuing.any() else None, 'draws': [], 'student_metrics': {}}
            for j in range(args.draws):
                item = {'u': noise[row, j].tolist(), 'independent_u': independent[row, j].tolist(), 'token_ids': token_lists[j], 'continuing': bool(continuing[j])}
                if row < 3 and j < 4:
                    item['text'] = tokenizer.decode(token_lists[j], skip_special_tokens=True)
                record['draws'].append(item)
            if row < 3:
                record['prompt'] = tokenizer.decode(data['prompts'][record['prompt_row']].tolist(), skip_special_tokens=True)
            for name, student, mode in students:
                supplied = u if mode == 'shared' else torch.full_like(u, .5)
                pred = student(h, supplied)
                mse = (pred.float() - feedback.float()).square().mean(-1)
                other = student(h, independent[row].to(device)) if mode == 'shared' else None
                unpaired = (other.float() - feedback.float()).square().mean(-1) if other is not None else None
                record['student_metrics'][name] = {'mode': mode, 'continuing_mse': float(mse[continuing].mean()) if continuing.any() else None, 'unpaired_continuing_mse': float(unpaired[continuing].mean()) if unpaired is not None and continuing.any() else None}
                for j in range(args.draws):
                    record['draws'][j].setdefault('student_mse', {})[name] = {'paired': float(mse[j]), 'unpaired': float(unpaired[j]) if unpaired is not None else None}
            records.append(record)
    summary = {'unique_contents': mean_stderr([r['unique_contents'] for r in records]), 'continuing_count': mean_stderr([r['continuing_count'] for r in records]), 'feedback_variance_continuing': mean_stderr([r['feedback_variance_continuing'] for r in records if r['feedback_variance_continuing'] is not None]), 'students': {}}
    for name, _, mode in students:
        metrics = [r['student_metrics'][name] for r in records]
        summary['students'][name] = {'continuing_mse': mean_stderr([m['continuing_mse'] for m in metrics if m['continuing_mse'] is not None]), 'paired_minus_unpaired': mean_stderr([m['continuing_mse'] - m['unpaired_continuing_mse'] for m in metrics if m['unpaired_continuing_mse'] is not None]) if mode == 'shared' else None}
    result = {'precision': precision, 'effective_precision': precision if device.type == 'cuda' else 'fp32', 'seed': args.seed, 'contexts': len(indices), 'draws_per_context': args.draws, 'teacher_hash': manifest['teacher_snapshot_sha256'], 'test_hash': manifest['split_sha256']['test'], 'note': 'Only complete non-EOS draws are used in student summary MSE; this conditions on continuation. Random numbers carry no intrinsic semantics; equal noise does not imply the same semantic future.', 'summary': summary, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
