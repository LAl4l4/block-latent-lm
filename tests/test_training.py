"""检查周期保存、中断保存及恢复；数据完全由测试临时生成。"""

import json
from pathlib import Path
import random
import signal
import sys

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from download_data import sha256
from latent_ar.control import GracefulStop
from latent_ar.model import LatentARModel, ModelConfig
import train


@pytest.fixture
def training_files(tmp_path):
    """生成几篇很小的故事，测试不依赖下载好的 TinyStories。"""
    data = tmp_path / "data"
    data.mkdir()
    texts = [
        f"A small cat found a red ball. It played all day. Story {i}."
        for i in range(12)
    ]
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        trainer=trainers.BpeTrainer(
            vocab_size=260,
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.save(str(data / "tokenizer.json"))

    for split in ["train", "dev", "test"]:
        ids = []
        offsets = [0]
        for text in texts:
            ids.extend(tokenizer.encode(text).ids + [2])
            offsets.append(len(ids))
        np.asarray(ids, dtype="<u4").tofile(data / f"{split}.bin")
        np.save(data / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.int64))

    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_ids": {"pad": 0, "bos": 1, "eos": 2},
        "tokenizer_sha256": sha256(data / "tokenizer.json"),
    }
    (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    config = {
        "name": "test",
        "model": {
            "encoder_dim": 8,
            "dim": 16,
            "layers": 1,
            "heads": 2,
            "ff_dim": 32,
            "decoder_dim": 16,
            "decoder_layers": 1,
            "decoder_heads": 2,
            "decoder_ff_dim": 32,
        },
        "training": {
            "seed": 7,
            "precision": "fp32",
            "cpu_threads": 1,
            "train_limit": None,
            "batch_size": 2,
            "grad_accum": 1,
            "min_prompt_tokens": 4,
            "max_prompt_tokens": 8,
            "max_target_tokens": 4,
            "max_steps": 101,
            "learning_rate": 0.001,
            "min_lr_ratio": 0.1,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "log_every": 1000,
            "eval_every": 1000,
            "eval_batches": 1,
            "save_every": 100,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return data, path


def run_training(monkeypatch, data, config, run_dir, *arguments):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config),
            "--data",
            str(data),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            *arguments,
        ],
    )
    train.main()


def test_autosave_at_100_and_at_end(training_files, tmp_path, monkeypatch):
    data, config = training_files
    saved_steps = []
    real_save = train.save_checkpoint

    def record_save(*arguments, **kwargs):
        saved_steps.append(arguments[4])
        return real_save(*arguments, **kwargs)

    monkeypatch.setattr(train, "save_checkpoint", record_save)
    run_dir = tmp_path / "periodic"
    run_training(monkeypatch, data, config, run_dir)
    assert saved_steps == [100, 101]
    checkpoint = torch.load(run_dir / "last.pt", weights_only=True)
    assert checkpoint["step"] == 101
    assert checkpoint["format_version"] == 2
    assert checkpoint["architecture"] == "latent_block_v2"
    assert checkpoint["model_config"]["decoder_type"] == "block_parallel"
    assert "optimizer" in checkpoint and "train_rng" in checkpoint


def test_interrupt_saves_completed_step_and_resumes_exactly(
    training_files, tmp_path, monkeypatch
):
    data, config = training_files
    reference_dir = tmp_path / "reference"
    interrupted_dir = tmp_path / "interrupted"
    run_training(
        monkeypatch,
        data,
        config,
        reference_dir,
        "--max-steps",
        "3",
        "--eval-batches",
        "2",
    )

    original_step = torch.optim.AdamW.step

    def interrupt_after_update(optimizer, *arguments, **kwargs):
        result = original_step(optimizer, *arguments, **kwargs)
        signal.raise_signal(signal.SIGINT)
        return result

    # 模拟用户在参数刚更新后按 Ctrl+C，而不是只测试一个布尔标志。
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_after_update)
        run_training(
            patch,
            data,
            config,
            interrupted_dir,
            "--max-steps",
            "3",
            "--eval-batches",
            "2",
        )

    interrupted = torch.load(interrupted_dir / "last.pt", weights_only=True)
    summary = json.loads((interrupted_dir / "summary.json").read_text())
    assert interrupted["step"] == 1
    assert interrupted["config"]["training"]["eval_batches"] == 2
    assert summary["interrupted"] is True

    run_training(
        monkeypatch,
        data,
        config,
        interrupted_dir,
        "--resume",
        str(interrupted_dir / "last.pt"),
    )
    reference = torch.load(reference_dir / "last.pt", weights_only=True)
    resumed = torch.load(interrupted_dir / "last.pt", weights_only=True)
    assert resumed["step"] == reference["step"] == 3
    assert resumed["tokens_seen"] == reference["tokens_seen"]
    assert resumed["train_rng"] == reference["train_rng"]
    for name in reference["model"]:
        assert torch.equal(reference["model"][name], resumed["model"][name]), name


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    model = LatentARModel(
        ModelConfig(
            vocab_size=32,
            encoder_dim=8,
            dim=16,
            layers=1,
            heads=2,
            ff_dim=32,
            decoder_dim=16,
            decoder_layers=1,
            decoder_heads=2,
            decoder_ff_dim=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous complete checkpoint")

    def failed_write(checkpoint, temporary_path):
        temporary_path.write_bytes(b"incomplete write")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", failed_write)
    with pytest.raises(OSError, match="disk write"):
        train.save_checkpoint(
            path, model, optimizer, scaler, 1, {}, "test", random.Random(7), 8
        )
    assert path.read_bytes() == b"previous complete checkpoint"


def test_stop_handler_restores_original_signal_handler():
    previous = signal.getsignal(signal.SIGINT)
    with GracefulStop() as stop:
        signal.raise_signal(signal.SIGINT)
        assert stop.requested
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) == previous
