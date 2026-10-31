"""两块一组前瞻的端到端训练与同预算同步对照。

从同一个 step10000 局部 GRU 教师出发，每个模式写入新的实验目录：

  sync   继续按原方式训练（每块等待真实反馈），作为同预算基线；
         评估时用 --student-init 指定的冻结 MSE 学生测 pair 条件下的 CE。
  state  组内第二块读取 Q(h)；Q 从 MSE 学生初始化，经语言模型损失训练。
  const  组内第二块读取一个可学习常量（不读取任何关于上一块的信息）。

--train all     更新主干、历史编码器、局部 GRU 和 Q（VAE 始终冻结）。
--train student 只更新 Q（教师冻结），即 "端到端训练的 P"。

dev 使用固定故事窗口，各模式可直接配对比较。CE 标签是语料真实文字。
"""

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoint_io import publish_checkpoint
from latent_ar.checkpoints import load_model
from latent_ar.control import GracefulStop
from latent_ar.data import StoryDataset
from latent_ar.feedback_student import FeedbackStudent
from latent_ar.pair_feedback import generate_pair_feedback
from latent_ar.pair_training import MODES, history_layout, make_constant, pair_statistics
from train import append_record, precision_context, to_device

SUM_KEYS = (
    "even_nll",
    "even_tokens",
    "odd_nll",
    "odd_tokens",
    "odd_first_nll",
    "odd_first_tokens",
    "even_first_nll",
    "even_first_tokens",
    "q_rel_error_sum",
    "q_norm_sum",
    "q_count",
)


def dataset_fingerprint(data):
    fingerprints = {
        f"{split}.{suffix}": sha256(data / f"{split}.{suffix}")
        for split in ("train", "dev")
        for suffix in ("bin", "offsets.npy")
    }
    fingerprints["metadata"] = sha256(data / "metadata.json")
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()


def summarize(sums, prefix):
    def ratio(a, b):
        return sums[a] / sums[b] if sums.get(b) else None

    return {
        f"{prefix}_ce": ratio("nll", "tokens"),
        f"{prefix}_even_ce": ratio("even_nll", "even_tokens"),
        f"{prefix}_odd_ce": ratio("odd_nll", "odd_tokens"),
        f"{prefix}_odd_first_ce": ratio("odd_first_nll", "odd_first_tokens"),
        f"{prefix}_even_first_ce": ratio("even_first_nll", "even_first_tokens"),
        f"{prefix}_q_rel_error": ratio("q_rel_error_sum", "q_norm_sum"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--train", choices=["all", "student"], default="all")
    p.add_argument("--init", type=Path, default=Path("runs/calm-repair-token/last.pt"))
    p.add_argument(
        "--student-init",
        type=Path,
        default=Path("runs/feedback-study/scan/n16384-h512-s2032/best.pt"),
    )
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--student-lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every) < 1:
        p.error("训练参数必须为正")
    if args.mode == "sync" and args.train == "student":
        p.error("sync 模式没有可单独训练的 Q")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"必须使用新的实验目录：{args.run_dir}")

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    data_rng = random.Random(args.seed)
    device = torch.device(args.device)
    source = torch.load(args.init, map_location="cpu", weights_only=True)
    if source["architecture"] != "latent_calm_token_v1":
        raise ValueError("需要局部 GRU 教师检查点 latent_calm_token_v1")
    prepared = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    if source["dataset_hash"] != dataset_fingerprint(args.data) or prepared[
        "tokenizer_sha256"
    ] != sha256(args.data / "tokenizer.json"):
        raise ValueError("数据或词表与初始化检查点不一致")
    model = load_model(source, device)
    model.codec.eval()
    init_hash = sha256(args.init)
    saved_student = torch.load(args.student_init, map_location="cpu", weights_only=True)
    student = FeedbackStudent.from_checkpoint(saved_student).to(device)
    if student.cfg.noise_dim or student.cfg.context_dim != model.cfg.dim:
        raise ValueError("需要确定性、维度匹配的 MLP 学生")
    student_hash = sha256(args.student_init)
    config = dict(source["config"])
    metadata = {k: source.get(k) for k in ("dataset_hash", "vae_source")}
    del source, saved_student

    tr = config["training"]
    train = StoryDataset(
        args.data, "train", tr["min_prompt_tokens"], tr["max_prompt_tokens"],
        tr["max_target_tokens"], model.cfg.pad_id,
    )
    dev = StoryDataset(
        args.data, "dev", tr["min_prompt_tokens"], tr["max_prompt_tokens"],
        tr["max_target_tokens"], model.cfg.pad_id,
    )
    autocast = precision_context(device, "bf16")

    if args.mode == "const":
        # 常量初始化为一批训练数据的真实反馈均值，等价于"最好的无信息猜测"。
        batch = to_device(train.batch(random.Random(args.seed + 7), 256), device)
        with torch.no_grad(), autocast():
            layout = history_layout(model, **batch)
            mask = layout["target_valid"]
            mean = layout["target_feedback"][mask].float().mean(0)
        make_constant(student)
        with torch.no_grad():
            student.output.bias.copy_(mean)

    train_model = args.train == "all"
    train_student = args.mode != "sync"
    model.requires_grad_(train_model)
    model.codec.requires_grad_(False)
    if not train_student:
        student.requires_grad_(False)
    groups = []
    if train_model:
        groups.append(
            {"params": [x for x in model.parameters() if x.requires_grad],
             "lr": args.lr, "weight_decay": 0.1, "base_lr": args.lr}
        )
    if train_student:
        groups.append(
            {"params": [x for x in student.parameters() if x.requires_grad],
             "lr": args.student_lr, "weight_decay": 0.01, "base_lr": args.student_lr}
        )
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.95), fused=device.type == "cuda"
    )
    trainable = sum(x.numel() for g in groups for x in g["params"])

    config["pair_lookahead"] = {
        "mode": args.mode,
        "train": args.train,
        "init": str(args.init),
        "init_sha256": init_hash,
        "student_init": str(args.student_init),
        "student_init_sha256": student_hash,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "student_lr": args.student_lr,
        "warmup": args.warmup,
        "seed": args.seed,
        "eval_batches": args.eval_batches,
        "eval_batch_size": args.eval_batch_size,
        "trainable_parameters": trainable,
        "student": asdict(student.cfg),
        "note": "Odd generated blocks (second in each pair) read Q(h_{t-1}) instead of "
        "the true feedback of block t-1; even blocks read all true feedback. sync "
        "trains the original objective; its pair metrics use the frozen student.",
    }
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.run_dir / "tokenizer.json"))
    (args.run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(config["pair_lookahead"], ensure_ascii=False), flush=True)
    train_mode = "sync" if args.mode == "sync" else args.mode

    @torch.inference_mode()
    def evaluate(step):
        was_model, was_student = model.training, student.training
        model.eval()
        student.eval()
        totals = {}
        # shuffled：同步前向但条件状态错配，给出每类块"历史全错"的 CE，
        # 汇总时用来把延迟代价换算成历史优势的比例。
        for name, mode, shuffle in (
            ("sync", "sync", False),
            ("pair", "state", False),
            ("shuffled", "sync", True),
        ):
            # const 学生也是 FeedbackStudent，pair 前向对两者通用。
            sums = {"nll": 0.0, "tokens": 0}
            for batch in dev.evaluation_batches(
                args.eval_batch_size, args.eval_batches, 1042
            ):
                batch = to_device(batch, device)
                with autocast():
                    values = pair_statistics(
                        model, student, **batch, mode=mode, shuffle_context=shuffle
                    )
                sums["nll"] += float(values["objective_sum"])
                sums["tokens"] += values["tokens"]
                for key in SUM_KEYS:
                    if key in values:
                        sums[key] = sums.get(key, 0) + values[key]
            totals.update(summarize(sums, name))
            totals[f"{name}_tokens"] = sums["tokens"]
        # 同步前向下的上下文优势，与教师训练日志的 context_advantage 同口径。
        right = wrong = count = 0.0
        rng = random.Random(1042)
        for _ in range(8):
            batch = to_device(dev.batch(rng, 16), device)
            with autocast():
                a = model.prior_statistics(**batch)
                b = model.prior_statistics(**batch, shuffle_context=True)
            right += float(a["objective_sum"])
            wrong += float(b["objective_sum"])
            count += a["tokens"]
        record = {
            "kind": "dev",
            "step": step,
            **totals,
            "pair_minus_sync_odd_ce": totals["pair_odd_ce"] - totals["sync_odd_ce"],
            "pair_minus_sync_odd_first_ce": totals["pair_odd_first_ce"]
            - totals["sync_odd_first_ce"],
            "context_advantage": (wrong - right) / count,
        }
        append_record(args.run_dir / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        temperature = model.cfg.token_temperature
        model.cfg.token_temperature = 0.0
        try:
            for text in (
                "Once upon a time, Lily found a little box.",
                "Tom wanted to play outside, but it was raining.",
            ):
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                prompt = torch.tensor([ids], device=device)
                lengths = torch.tensor([len(ids)], device=device)
                with autocast():
                    sync_out = model.generate(
                        prompt, lengths, max_new_tokens=64,
                        generator=torch.Generator(device=device).manual_seed(0),
                    )
                    pair_out = generate_pair_feedback(
                        model, student, prompt, lengths, max_new_tokens=64,
                        generator=torch.Generator(device=device).manual_seed(0),
                        parallel="batch",
                    )
                append_record(
                    args.run_dir / "samples.jsonl",
                    {
                        "step": step,
                        "prompt": text,
                        "greedy_sync": tokenizer.decode(sync_out[0].tolist()),
                        "greedy_pair": tokenizer.decode(pair_out[0].tolist()),
                    },
                )
        finally:
            model.cfg.token_temperature = temperature
        model.train(was_model)
        student.train(was_student)
        model.codec.eval()

    def save(step, seen):
        checkpoint = {
            "architecture": model.architecture,
            "model_config": asdict(model.cfg),
            "model": model.state_dict(),
            "step": step,
            "config": config,
            "device_type": device.type,
            "tokens_seen": seen,
            "pair_lookahead_student": student.checkpoint(),
            **metadata,
        }
        temporary = args.run_dir / "last.tmp"
        torch.save(checkpoint, temporary)
        publish_checkpoint(temporary, args.run_dir / "last.pt", step)
        student_ckpt = student.checkpoint() | {
            "step": step,
            "pair_lookahead_mode": args.mode,
            "teacher_checkpoint": str(args.run_dir / "last.pt"),
        }
        temporary = args.run_dir / "student.tmp"
        torch.save(student_ckpt, temporary)
        publish_checkpoint(temporary, args.run_dir / "student.pt", step)

    evaluate(0)
    # 冻结靠 requires_grad；主干、E、GRU 都没有 dropout，训练模式与 eval 等价。
    # cuDNN 的 GRU 只能在训练模式下反传，--train student 也需要梯度穿过它到 Q。
    model.train()
    model.codec.eval()
    student.train(train_student)
    seen = 0
    with GracefulStop() as stop:
        for step in range(1, args.steps + 1):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batches = [
                to_device(train.batch(data_rng, args.batch_size), device)
                for _ in range(args.grad_accum)
            ]
            tokens = sum(int(b["labels"].ne(-100).sum()) for b in batches)
            loss_value, odd_nll, odd_tokens = 0.0, 0.0, 0
            for batch in batches:
                with autocast():
                    values = pair_statistics(model, student, **batch, mode=train_mode)
                    loss = values["objective_sum"] / tokens
                loss.backward()
                loss_value += float(loss.detach())
                odd_nll += values["odd_nll"]
                odd_tokens += values["odd_tokens"]
            params = [x for g in groups for x in g["params"]]
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            scale = min(1.0, step / max(1, args.warmup))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            seen += tokens
            record = {
                "kind": "train",
                "step": step,
                "loss": loss_value,
                "odd_ce": odd_nll / max(1, odd_tokens),
                "tokens": tokens,
                "grad_norm": float(norm),
                "seconds": time.perf_counter() - begin,
            }
            append_record(args.run_dir / "metrics.jsonl", record)
            if step % 50 == 0 or step == 1:
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps or stop.requested:
                save(step, seen)
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
            if stop.requested:
                break
    (args.run_dir / "DONE").write_text(str(step), encoding="utf-8")


if __name__ == "__main__":
    main()
