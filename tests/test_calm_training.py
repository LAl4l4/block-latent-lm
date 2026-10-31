"""使用临时小数据验证 VAE 与 CALM 的 CPU 保存恢复链路。"""

import json
import sys

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from download_data import sha256
from latent_ar.calm_training import run
import generate


def write_inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"word{i}": i + 4 for i in range(28)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(data / "tokenizer.json"))
    for split in ("train", "dev", "test"):
        all_ids, offsets = [], [0]
        for row in range(16):
            all_ids.extend([4 + (row + i) % 28 for i in range(32)] + [2])
            offsets.append(len(all_ids))
        np.asarray(all_ids, dtype="<u4").tofile(data / f"{split}.bin")
        np.save(data / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.int64))
    (data / "metadata.json").write_text(
        json.dumps(
            {
                "vocab_size": 32,
                "special_ids": {"pad": 0, "bos": 1, "eos": 2},
                "tokenizer_sha256": sha256(data / "tokenizer.json"),
            }
        ),
        encoding="utf-8",
    )
    model = {
        "chunk_size": 4,
        "encoder_dim": 16,
        "dim": 32,
        "layers": 1,
        "heads": 4,
        "ff_dim": 64,
        "decoder_dim": 16,
        "decoder_layers": 1,
        "decoder_heads": 2,
        "decoder_ff_dim": 32,
    }
    training = {
        "seed": 7,
        "precision": "fp32",
        "cpu_threads": 1,
        "batch_size": 2,
        "grad_accum": 1,
        "min_prompt_tokens": 4,
        "max_prompt_tokens": 8,
        "max_target_tokens": 8,
        "max_steps": 2,
        "learning_rate": 0.001,
        "min_lr_ratio": 0.1,
        "warmup_steps": 1,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "log_every": 1,
        "eval_every": 2,
        "eval_batches": 1,
        "save_every": 1,
    }
    config = tmp_path / "vae.json"
    config.write_text(
        json.dumps({"name": "test-vae", "model": model, "training": training}),
        encoding="utf-8",
    )
    return data, config, model, training


def start(monkeypatch, stage, data, config, directory, *options):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--device",
            "cpu",
            "--data",
            str(data),
            "--config",
            str(config),
            "--run-dir",
            str(directory),
            *options,
        ],
    )
    run(stage)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_vae_resume_exact_and_calm_save_resume_generate(tmp_path, monkeypatch, capsys):
    data, vae_config, model_config, training = write_inputs(tmp_path)
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    start(monkeypatch, "vae", data, vae_config, reference)
    start(monkeypatch, "vae", data, vae_config, resumed, "--stop-after", "1")
    start(
        monkeypatch,
        "vae",
        data,
        vae_config,
        resumed,
        "--resume",
        str(resumed / "last.pt"),
    )
    left = torch.load(reference / "last.pt", map_location="cpu", weights_only=True)
    right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    assert left["step"] == right["step"] == 2
    for key in (
        "model",
        "optimizer",
        "train_rng",
        "torch_rng",
        "noise_rng",
        "tokens_seen",
    ):
        assert_same(left[key], right[key])

    calm_config = tmp_path / "calm.json"
    calm_model = dict(
        model_config,
        noise_dim=8,
        energy_candidates=2,
        target_samples=2,
        energy_layers=1,
        energy_batch_blocks=4,
    )
    calm_config.write_text(
        json.dumps(
            {
                "name": "test-calm",
                "vae_checkpoint": str(reference / "last.pt"),
                "quality_gate": {
                    "token_accuracy": 0.0,
                    "block_accuracy": 0.0,
                    "eval_blocks": 4,
                },
                "model": calm_model,
                "training": training,
            }
        ),
        encoding="utf-8",
    )
    calm_dir = tmp_path / "calm"
    start(monkeypatch, "calm", data, calm_config, calm_dir, "--stop-after", "1")
    first = torch.load(calm_dir / "last.pt", map_location="cpu", weights_only=True)
    assert first["step"] == 1 and first["vae_source"]["checkpoint"]
    start(
        monkeypatch,
        "calm",
        data,
        calm_config,
        calm_dir,
        "--resume",
        str(calm_dir / "last.pt"),
    )
    checkpoint = torch.load(calm_dir / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2 and checkpoint["architecture"] == "latent_calm_v1"
    calm_reference = tmp_path / "calm-reference"
    start(monkeypatch, "calm", data, calm_config, calm_reference)
    uninterrupted = torch.load(calm_reference / "last.pt", map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "train_rng", "torch_rng", "noise_rng", "tokens_seen"):
        assert_same(checkpoint[key], uninterrupted[key])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--checkpoint",
            str(calm_dir / "last.pt"),
            "--prompt",
            "word1 word2",
            "--max-new-tokens",
            "4",
            "--device",
            "cpu",
        ],
    )
    generate.main()
    assert "word1 word2" in capsys.readouterr().out
