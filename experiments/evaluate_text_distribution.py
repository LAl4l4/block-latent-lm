"""固定 dev 上比较隐式/有限混合 latent 的文字分布，不把单样本 CE 当 PPL。"""

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoints import ENERGY_ARCHITECTURE, MIXTURE_ARCHITECTURE, load_model
from latent_ar.data import StoryDataset
from train import precision_context, select_device, to_device


def brier_ngrams(samples, labels):
    """独立生成候选 [blocks,N,4] 的无偏 n-gram Brier reward，越大越好。

    同块内所有连续 n-gram 起点都计分；不跨块。所有非对角独立对估计碰撞率。
    """
    count, draws, width = samples.shape
    if draws < 2:
        raise ValueError("Brier estimation requires two or more draws")
    result = {}
    for n in range(1, width + 1):
        truth, collision, units = 0.0, 0.0, 0
        for start in range(width - n + 1):
            valid = (labels[:, start : start + n] != -100).all(-1)
            values = samples[valid, :, start : start + n]
            reference = labels[valid, start : start + n]
            hits = (values == reference[:, None]).all(-1).float()
            same = (values[:, :, None] == values[:, None, :]).all(-1).float()
            truth += float(hits.sum()) / draws
            collision += float(same.sum() - len(values) * draws) / (draws * (draws - 1))
            units += len(values)
        result[str(n)] = {
            "reward": (2 * truth - collision) / units if units else None,
            "reference_match": truth / units if units else None,
            "collision": collision / units if units else None,
            "units": units,
        }
    return result


@torch.inference_mode()
def evaluate(checkpoint_path, batch, draws, seed, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["dataset_hash"] != sha256(Path("data/prepared/metadata.json")):
        raise ValueError("Checkpoint dataset mismatch")
    model = load_model(checkpoint, device).eval()
    autocast = precision_context(device, checkpoint["config"]["training"]["precision"])
    rng = torch.Generator(device=device).manual_seed(seed)
    with autocast():
        contexts, targets, valid = model.teacher_contexts(
            batch["prompt"], batch["prompt_lengths"], batch["labels"]
        )
    contexts = contexts[valid]
    labels = F.pad(
        batch["labels"],
        (0, targets.shape[1] * model.cfg.chunk_size - batch["labels"].shape[1]),
        value=-100,
    ).reshape(*valid.shape, model.cfg.chunk_size)[valid]
    generated = {"greedy_decoder": [], "softmax_decoder": []}
    for start in range(0, len(contexts), 32):
        context = contexts[start : start + 32, None].expand(-1, draws, -1)
        with autocast():
            latent = model.sample_latent(context, generator=rng)
            logits = model.codec.decoder(latent).float()
        # 正式生成禁止 PAD/BOS，评价同步屏蔽；无 top-k/温度修改。
        logits[..., model.cfg.pad_id] = -torch.inf
        logits[..., model.cfg.bos_id] = -torch.inf
        generated["greedy_decoder"].append(logits.argmax(-1))
        sampled = torch.multinomial(
            logits.softmax(-1).reshape(-1, model.cfg.vocab_size), 1, generator=rng
        ).reshape(*logits.shape[:-1])
        generated["softmax_decoder"].append(sampled)
    result = {
        name: brier_ngrams(torch.cat(values), labels)
        for name, values in generated.items()
    }
    tokenizer = Tokenizer.from_file(str(checkpoint_path.parent / "tokenizer.json"))
    result["free_text"] = []
    for row in range(4):
        for seed_value in [42, 43]:
            with autocast():
                output = model.generate(
                    batch["prompt"][row : row + 1],
                    batch["prompt_lengths"][row : row + 1],
                    64,
                    temperature=0.0,
                    top_k=0,
                    generator=torch.Generator(device=device).manual_seed(seed_value),
                )
            ids = output[0].tolist()
            if model.cfg.eos_id in ids:
                ids = ids[: ids.index(model.cfg.eos_id)]
            result["free_text"].append(
                {
                    "row": row,
                    "seed": seed_value,
                    "prompt": tokenizer.decode(
                        batch["prompt"][row, : batch["prompt_lengths"][row]].tolist()
                    ),
                    "continuation": tokenizer.decode(ids),
                }
            )
    return {"checkpoint": str(checkpoint_path), "step": checkpoint["step"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/text-distribution-comparison.json")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10042)
    args = parser.parse_args()
    if args.batch_size < 4 or args.draws < 2:
        parser.error("batch-size >= 4 and draws >= 2 required")
    torch.set_num_threads(4)
    device = select_device("auto")
    dataset = StoryDataset(Path("data/prepared"), "dev", 16, 128, 64, 0)
    batch = to_device(dataset.batch(random.Random(args.seed), args.batch_size), device)
    reports = []
    for path in args.checkpoints:
        result = evaluate(path, batch, args.draws, args.seed + 1, device)
        reports.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "free_text"}),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "draws": args.draws,
                "batch_size": args.batch_size,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
