"""同一数据上的普通 token AR 对照，检验数据/优化与 latent 表示瓶颈。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.control import GracefulStop
from latent_ar.data import StoryDataset
from latent_ar.model import LatentTransformer, ModelConfig
from train import append_record, precision_context, select_device


class TokenAR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.backbone = LatentTransformer(cfg)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, valid, positions=None, caches=None):
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device).expand(
                len(ids), -1
            )
        hidden, caches = self.backbone(self.embedding(ids), positions, valid, caches)
        return self.output(hidden), caches

    @torch.inference_mode()
    def generate(self, ids, generator, count=64, temperature=0.8):
        valid = torch.ones_like(ids, dtype=torch.bool)
        logits, caches = self(ids, valid)
        output = []
        for step in range(count):
            scores = logits[:, -1].float() / temperature
            scores[:, self.cfg.pad_id] = -torch.inf
            scores[:, self.cfg.bos_id] = -torch.inf
            # 未截断完整分类分布；同一 seed 可复查。
            token = torch.multinomial(scores.softmax(-1), 1, generator=generator)
            output.append(int(token))
            if int(token) == self.cfg.eos_id:
                break
            valid = torch.cat(
                [valid, torch.ones(1, 1, dtype=torch.bool, device=ids.device)], -1
            )
            positions = torch.full(
                (1, 1), ids.shape[1] + step, device=ids.device, dtype=torch.long
            )
            logits, caches = self(token, valid, positions, caches)
        return output


def packed_batch(dataset, rng, count, device):
    raw = dataset.batch(rng, count)
    rows, labels = [], []
    for prompt, length, target in zip(
        raw["prompt"], raw["prompt_lengths"], raw["labels"]
    ):
        target = target[target != -100]
        sequence = torch.cat([prompt[:length], target])
        rows.append(sequence[:-1])
        label = sequence[1:].clone()
        label[: int(length) - 1] = -100
        labels.append(label)
    width = max(map(len, rows))
    ids = torch.full((count, width), dataset.pad_id, dtype=torch.long)
    target = torch.full((count, width), -100, dtype=torch.long)
    valid = torch.zeros(count, width, dtype=torch.bool)
    for i, (row, label) in enumerate(zip(rows, labels)):
        ids[i, : len(row)], target[i, : len(row)], valid[i, : len(row)] = (
            row,
            label,
            True,
        )
    return tuple(x.to(device) for x in (ids, target, valid))


@torch.inference_mode()
def evaluate(model, dataset, autocast, device):
    model.eval()
    total, tokens = 0.0, 0
    rng = random.Random(10042)
    for _ in range(4):
        ids, labels, valid = packed_batch(dataset, rng, 32, device)
        with autocast():
            logits, _ = model(ids, valid)
            loss = F.cross_entropy(
                logits.float().flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        total += float(loss)
        tokens += int((labels != -100).sum())
    model.train()
    return total / tokens


def save_samples(model, tokenizer, device, autocast, run_dir, step, prompts, seeds):
    model.eval()
    for text in prompts:
        for seed in seeds:
            ids = torch.tensor(
                [tokenizer.encode(text, add_special_tokens=False).ids], device=device
            )
            if ids.shape[1] == 0:
                raise ValueError("Prompt must contain tokens")
            ids = ids[:, -128:]
            with autocast():
                output = model.generate(
                    ids, torch.Generator(device=device).manual_seed(seed)
                )
            record = {
                "step": step,
                "seed": seed,
                "prompt": text,
                "continuation": tokenizer.decode(output),
            }
            append_record(run_dir / "samples.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    args.run_dir = args.run_dir or (
        args.resume.parent if args.resume else Path("runs/token-ar-control")
    )
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = random.Random(42)
    device = select_device("auto")
    autocast = precision_context(device, "bf16")
    metadata = json.loads(
        Path("data/prepared/metadata.json").read_text(encoding="utf-8")
    )
    cfg = ModelConfig(
        vocab_size=metadata["vocab_size"], dim=256, layers=6, heads=8, ff_dim=768
    )
    model = TokenAR(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )
    first = 1
    dataset_hash = sha256(Path("data/prepared/metadata.json"))
    if args.resume:
        source = torch.load(args.resume, map_location="cpu", weights_only=True)
        if (
            source["dataset_hash"] != dataset_hash
            or source["architecture"] != "token_ar_control_v1"
        ):
            raise ValueError("Checkpoint/data mismatch")
        model.load_state_dict(source["model"], strict=True)
        optimizer.load_state_dict(source["optimizer"])
        rng.setstate(source["data_rng"])
        torch.set_rng_state(source["torch_rng"].cpu())
        if source["device_type"] != device.type:
            raise ValueError("Exact resume requires the same device type")
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in source["cuda_rng"]])
        first = source["step"] + 1
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("Use a new experiment directory")
    if not args.eval_only and args.steps < first:
        raise ValueError("No remaining steps")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: StoryDataset(Path("data/prepared"), split, 16, 128, 64, cfg.pad_id)
        for split in ["train", "dev"]
    }
    print(f"[parameters] {sum(p.numel() for p in model.parameters()):,}", flush=True)
    tokenizer = Tokenizer.from_file("data/prepared/tokenizer.json")
    prompts = args.prompt or [
        "Once upon a time, a little girl found a strange box.",
        "Tom and Lily went to the park. They saw a big dog.",
    ]
    if args.eval_only:
        print(f"[dev] CE {evaluate(model, datasets['dev'], autocast, device):.4f}")
        save_samples(
            model,
            tokenizer,
            device,
            autocast,
            args.run_dir,
            first - 1,
            prompts,
            args.seeds,
        )
        return
    if not args.resume:
        ce = evaluate(model, datasets["dev"], autocast, device)
        append_record(
            args.run_dir / "metrics.jsonl", {"kind": "dev", "step": 0, "ce": ce}
        )
        print(f"[dev 0] CE {ce:.4f}", flush=True)
    with GracefulStop() as stop:
        for step in range(first, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batches = [
                packed_batch(datasets["train"], rng, 32, device) for _ in range(2)
            ]
            tokens = sum(int((y != -100).sum()) for _, y, _ in batches)
            total = 0.0
            for ids, labels, valid in batches:
                with autocast():
                    logits, _ = model(ids, valid)
                    loss_sum = F.cross_entropy(
                        logits.float().flatten(0, 1),
                        labels.flatten(),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    loss = loss_sum / tokens
                loss.backward()
                total += float(loss_sum.detach())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            for group in optimizer.param_groups:
                group["lr"] = 0.0003 * min(1.0, step / 100)
            optimizer.step()
            append_record(
                args.run_dir / "metrics.jsonl",
                {"kind": "train", "step": step, "ce": total / tokens},
            )
            if step % 50 == 0 or step == first:
                print(f"[train {step}] CE {total/tokens:.4f}", flush=True)
            if (step % 250 == 0 or step == args.steps) and not stop.requested:
                ce = evaluate(model, datasets["dev"], autocast, device)
                append_record(
                    args.run_dir / "metrics.jsonl",
                    {"kind": "dev", "step": step, "ce": ce},
                )
                print(f"[dev {step}] CE {ce:.4f}", flush=True)
            if step % 100 == 0 or step == args.steps or stop.requested:
                checkpoint = {
                    "architecture": "token_ar_control_v1",
                    "model_config": asdict(cfg),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "data_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "device_type": device.type,
                    "dataset_hash": dataset_hash,
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                    ),
                }
                temporary = args.run_dir / "last.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(args.run_dir / "last.pt")
            if stop.requested:
                break
    save_samples(
        model, tokenizer, device, autocast, args.run_dir, step, prompts, args.seeds
    )


if __name__ == "__main__":
    main()
