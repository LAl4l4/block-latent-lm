"""独立短测训练批量的显存和吞吐；不读取或覆盖正式训练的检查点。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from latent_ar.data import StoryDataset, collate
from latent_ar.model import LatentARModel, ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/batch-benchmark.json")
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.steps < 1:
        parser.error("CUDA and at least one measurement step are required")
    if args.grad_accum < 1 or min(args.batch_sizes) < 1:
        parser.error("Batch sizes and grad-accum must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    cfg = ModelConfig(
        vocab_size=metadata["vocab_size"],
        **{
            f"{name}_id": metadata["special_ids"][name]
            for name in ["pad", "bos", "eos"]
        },
    )
    dataset = StoryDataset(args.data, "train", 128, 128, 64, cfg.pad_id)
    required_examples = max(args.batch_sizes)
    examples = []
    # 固定取足够长的真实故事，保证测到最大 prompt/target 长度的开销。
    for index in range(dataset.size):
        story = dataset.tokens[dataset.offsets[index] : dataset.offsets[index + 1]]
        if len(story) >= 192:
            examples.append(
                (story[:128].astype("int64"), story[128:192].astype("int64"))
            )
            if len(examples) == required_examples:
                break
    if len(examples) < required_examples:
        raise RuntimeError(f"Need {required_examples} stories with at least 192 tokens")
    model = LatentARModel(cfg).cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True
    )
    results = []
    for batch_size in args.batch_sizes:
        accumulation = args.grad_accum
        batch = {
            k: v.cuda() for k, v in collate(examples[:batch_size], cfg.pad_id).items()
        }

        def update():
            """执行完整的累计梯度与参数更新，避免只测前向低估显存。"""
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(
                        batch["prompt"],
                        batch["prompt_lengths"],
                        batch["labels"].shape[1],
                    )
                    loss = F.cross_entropy(
                        logits.float().flatten(0, 1), batch["labels"].flatten()
                    )
                (loss / accumulation).backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            return float(loss.detach()), float(norm)

        # 第一遍预热 CUDA 内核并建立优化器状态，不计入平均耗时。
        update()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            ce, norm = update()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / args.steps
        free, total = torch.cuda.mem_get_info()
        record = {
            "batch_size": batch_size,
            "grad_accum": accumulation,
            "effective_batch": batch_size * accumulation,
            "prompt_tokens": 128,
            "target_tokens": 64,
            "latent_steps": 16,
            "seconds_per_update": seconds,
            "target_tokens_per_second": batch_size * accumulation * 64 / seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "device_free_mib_after": free / 2**20,
            "device_total_mib": total / 2**20,
            "last_ce": ce,
            "last_grad_norm": norm,
        }
        results.append(record)
        print(json.dumps(record), flush=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    report = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "model_config": asdict(cfg),
        "measurement_steps": args.steps,
        "note": "Full-length synthetic crops from real stories; concurrent workloads can affect timings.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
