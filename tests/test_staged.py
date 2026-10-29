"""分阶段训练的表示、因果边界、采样与恢复检查。"""

from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys

import pytest
import torch
import torch.nn.functional as F

from latent_ar.checkpoints import checkpoint_architecture, load_model
from latent_ar.codec import BlockCodec, reconstruction_statistics
from latent_ar.model import ModelConfig
from latent_ar.staged import LatentPriorModel, StagedModelConfig
from latent_ar.staged_training import run_stage
from test_training import training_files


def small_config():
    return StagedModelConfig(
        vocab_size=32,
        encoder_dim=16,
        dim=32,
        layers=1,
        heads=4,
        ff_dim=64,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ff_dim=32,
    )


@pytest.fixture
def prior():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return LatentPriorModel(small_config())


def examples():
    prompt = torch.tensor([[4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 0, 0, 0, 0]])
    lengths = torch.tensor([7, 3])
    labels = torch.tensor(
        [
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 2],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 2, -100, -100],
        ]
    )
    return prompt, lengths, labels


def test_codec_learns_real_four_token_reconstruction():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    cfg = ModelConfig(
        vocab_size=32,
        encoder_dim=16,
        dim=64,
        layers=1,
        heads=4,
        ff_dim=64,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ff_dim=64,
    )
    model = BlockCodec(cfg)
    assert model.encoder.embedding.weight is model.decoder.output.weight
    ids = torch.randint(4, 32, (96, 4))
    lengths = torch.full((96,), 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    for _ in range(250):
        optimizer.zero_grad()
        logits = model(ids, lengths)
        loss = F.cross_entropy(logits.flatten(0, 1), ids.flatten())
        loss.backward()
        optimizer.step()
    predictions = model(ids, lengths).argmax(-1)
    assert float((predictions == ids).float().mean()) >= 0.99
    assert float((predictions == ids).all(-1).float().mean()) >= 0.95


def test_short_blocks_do_not_inflate_full_four_token_accuracy():
    logits = torch.zeros(2, 4, 8)
    logits[..., 4] = 10
    labels = torch.tensor([[4, 5, 6, 7], [4, -100, -100, -100]])
    stats = reconstruction_statistics(logits, labels)
    assert stats["blocks"] == 2 and stats["correct_blocks"] == 1
    assert stats["full_blocks"] == 1 and stats["correct_full_blocks"] == 0


def test_staged_training_can_memorize_and_freely_continue_fixed_sequences():
    """先拟合小块，再拟合确定的三块续写，最后必须能自主回灌还原。"""
    torch.set_num_threads(1)
    torch.manual_seed(101)
    cfg = small_config()
    codec = BlockCodec(cfg)
    stories = torch.randint(4, cfg.vocab_size, (4, 4, cfg.chunk_size))
    blocks = stories.reshape(-1, cfg.chunk_size)
    lengths = torch.full((16,), cfg.chunk_size)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=0.01, weight_decay=0)
    for _ in range(250):
        optimizer.zero_grad()
        logits = codec(blocks, lengths)
        F.cross_entropy(logits.flatten(0, 1), blocks.flatten()).backward()
        optimizer.step()
    assert torch.equal(codec(blocks, lengths).argmax(-1), blocks)

    model = LatentPriorModel(cfg)
    model.codec.load_state_dict(codec.state_dict())
    with torch.no_grad():
        bank, _ = model.codec.encode(blocks, lengths)
        model.set_latent_statistics(bank)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.003, weight_decay=0
    )
    prompt, targets = stories[:, 0], stories[:, 1:].reshape(4, -1)
    prompt_lengths = torch.full((4,), cfg.chunk_size)
    for _ in range(300):
        optimizer.zero_grad()
        stats = model.prior_statistics(prompt, prompt_lengths, targets)
        loss = (
            stats["latent_mse_sum"] / stats["blocks"]
            + stats["token_ce_sum"] / stats["tokens"]
        )
        loss.backward()
        optimizer.step()
    # 真正推理只能收到 prompt，不能读取 targets 或真实后续 latent。
    actual = model.generate(prompt, prompt_lengths, max_new_tokens=12)
    assert torch.equal(actual, targets)


def test_prior_keeps_codec_frozen_and_has_gradients(prior):
    prior.train()
    assert not prior.codec.training
    prompt, lengths, labels = examples()
    stats = prior.prior_statistics(prompt, lengths, labels)
    assert stats["blocks"] == 6
    # 单独检查 CE 能穿过冻结 decoder，到达主干；否则双重监督名存实亡。
    stats["token_ce_sum"].backward(retain_graph=True)
    assert prior.latent_head.weight.grad.abs().sum() > 0
    prior.zero_grad(set_to_none=True)
    (
        stats["latent_mse_sum"] / stats["blocks"]
        + stats["token_ce_sum"] / stats["tokens"]
    ).backward()
    assert all(not p.requires_grad and p.grad is None for p in prior.codec.parameters())
    for module in [prior.backbone, prior.latent_head]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_teacher_context_cannot_see_current_or_future_target(prior):
    prompt, lengths, labels = examples()
    first, _, _ = prior.teacher_contexts(prompt, lengths, labels)
    changed = labels.clone()
    changed[:, 4:8] = 25
    second, _, _ = prior.teacher_contexts(prompt, lengths, changed)
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert not torch.allclose(first[:, 2], second[:, 2])


def test_parallel_teacher_context_matches_cached_true_history(prior):
    prompt, lengths, labels = examples()
    contexts, targets, _ = prior.teacher_contexts(prompt, lengths, labels)
    encoded, valid = prior.codec.encode(prompt, lengths)
    sizes = valid.sum(-1)
    positions = torch.arange(encoded.shape[1]).expand(2, -1)
    hidden, caches = prior.backbone(encoded, positions, valid)
    current = hidden[torch.arange(2), sizes - 1]
    torch.testing.assert_close(contexts[:, 0], current, atol=2e-6, rtol=2e-5)
    for index in range(2):
        valid = torch.cat([valid, torch.ones(2, 1, dtype=torch.bool)], dim=1)
        hidden, caches = prior.backbone(
            targets[:, index : index + 1], (sizes + index)[:, None], valid, caches
        )
        torch.testing.assert_close(
            contexts[:, index + 1], hidden[:, 0], atol=2e-6, rtol=2e-5
        )


def test_prior_padding_has_no_loss_gradient(prior, monkeypatch):
    contexts = torch.randn(2, 3, prior.cfg.dim, requires_grad=True)
    targets = torch.randn(2, 3, prior.cfg.dim)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    labels = torch.tensor(
        [
            [4, 5, 6, 7, 8, 9, 10, 2, -100, -100, -100, -100],
            [4, 5, 2, -100, -100, -100, -100, -100, -100, -100, -100, -100],
        ]
    )
    monkeypatch.setattr(
        prior, "teacher_contexts", lambda *args: (contexts, targets, valid)
    )
    stats = prior.prior_statistics(None, None, labels)
    assert stats["blocks"] == 3 and stats["tokens"] == 11
    (stats["latent_mse_sum"] + stats["token_ce_sum"]).backward()
    assert torch.count_nonzero(contexts.grad[~valid]) == 0
    assert contexts.grad[valid].abs().sum() > 0


def test_teacher_and_free_first_prediction_agree(prior):
    prompt, lengths, labels = examples()
    contexts, _, _ = prior.teacher_contexts(prompt, lengths, labels)
    expected = prior.denormalize(prior.latent_head(contexts[:, 0]))
    actual = prior.rollout(prompt, lengths, 3)[:, 0]
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-5)


def test_free_rollout_feeds_back_predicted_latent(prior, monkeypatch):
    prompt, lengths, _ = examples()
    predictions = torch.randn(2, 3, prior.cfg.dim)
    next_prediction = iter(predictions.unbind(1))
    monkeypatch.setattr(
        prior.latent_head, "forward", lambda context: next(next_prediction)
    )
    seen = []
    hook = prior.backbone.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    generated = prior.rollout(prompt, lengths, 3)
    hook.remove()
    torch.testing.assert_close(generated, prior.denormalize(predictions))
    torch.testing.assert_close(seen[1][:, 0], generated[:, 0])
    torch.testing.assert_close(seen[2][:, 0], generated[:, 1])


def test_latent_prediction_is_deterministic_and_independent_of_token_sampling(prior):
    prompt, lengths, _ = examples()
    captured = []
    method = prior.rollout

    def capture(*args, **kwargs):
        result = method(*args, **kwargs)
        captured.append(result.detach().clone())
        return result

    prior.rollout = capture
    prior.generate(
        prompt,
        lengths,
        max_new_tokens=8,
        temperature=0,
        generator=torch.Generator().manual_seed(42),
    )
    prior.generate(
        prompt,
        lengths,
        max_new_tokens=8,
        temperature=0.8,
        generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(captured[0], captured[1])
    prior.generate(
        prompt, lengths, max_new_tokens=8, generator=torch.Generator().manual_seed(43)
    )
    torch.testing.assert_close(captured[0], captured[2])


def test_staged_checkpoint_loader_preserves_shared_codec_and_normalization(prior):
    prior.set_latent_statistics(torch.randn(32, prior.cfg.dim) * 2 + 3)
    checkpoint = {
        "architecture": prior.architecture,
        "model_config": asdict(prior.cfg),
        "model": prior.state_dict(),
    }
    loaded = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(loaded, LatentPriorModel)
    assert loaded.codec.encoder.embedding.weight is loaded.codec.decoder.output.weight
    for key, value in prior.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    assert all(not p.requires_grad for p in loaded.codec.parameters())


@pytest.fixture
def stage_files(training_files, tmp_path):
    data, old_config = training_files
    base = json.loads(old_config.read_text())
    base["name"] = "codec-test"
    base["model"].update(encoder_dim=16, dim=32, heads=4, ff_dim=64)
    base["training"].update(max_steps=3, eval_every=2, eval_batches=1, save_every=2)
    codec_config = tmp_path / "codec.json"
    codec_config.write_text(json.dumps(base), encoding="utf-8")
    prior_config = tmp_path / "prior.json"
    settings = {
        "name": "prior-test",
        "codec_checkpoint": str(tmp_path / "codec" / "last.pt"),
        # 只为流程测试降低阈值，不把未训练 codec 当成重建质量达标。
        "quality_gate": {"token_accuracy": 0, "block_accuracy": 0, "eval_blocks": 8},
        "latent_stats_blocks": 8,
        "model": {
            "layers": 1,
            "heads": 4,
            "ff_dim": 64,
        },
        "training": base["training"],
    }
    prior_config.write_text(json.dumps(settings), encoding="utf-8")
    return data, codec_config, prior_config


def launch(monkeypatch, stage, data, config, run_dir, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"train_{stage}.py",
            "--config",
            str(config),
            "--data",
            str(data),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            *extra,
        ],
    )
    run_stage(stage)


@pytest.mark.parametrize(
    "stage",
    ["codec", "prior", "energy", "mixture", "latent_mse", "denoising", "token_head"],
)
def test_each_stage_interrupt_resume_matches_uninterrupted(
    stage, stage_files, tmp_path, monkeypatch
):
    data, codec_config, prior_config = stage_files
    if stage in {"energy", "mixture", "latent_mse", "denoising", "token_head"}:
        settings = json.loads(prior_config.read_text())
        settings["prior_type"] = "ar" if stage == "latent_mse" else stage
        if stage == "energy":
            settings["model"].update(
                noise_dim=8, energy_hidden_dim=32, energy_layers=1, energy_candidates=3
            )
        elif stage == "mixture":
            settings["model"].update(
                mixture_components=3,
                mixture_hidden_dim=32,
                mixture_layers=1,
                mixture_train_blocks=2,
            )
        elif stage == "token_head":
            settings["model"].update(
                token_head_dim=16, token_temperature=0.8, token_top_k=16
            )
        else:
            settings["training"].update(latent_mse_weight=1.0, token_ce_weight=0.0)
            if stage == "denoising":
                settings["model"].update(
                    denoise_hidden_dim=32,
                    denoise_layers=1,
                    denoise_steps=2,
                    context_dropout=0.1,
                    denoise_guidance=1.0,
                )
        prior_config.write_text(json.dumps(settings), encoding="utf-8")
        stage = "prior"
    launch(monkeypatch, "codec", data, codec_config, tmp_path / "codec")
    config = codec_config if stage == "codec" else prior_config
    reference_dir = tmp_path / f"{stage}-reference"
    interrupted_dir = tmp_path / f"{stage}-interrupted"
    launch(monkeypatch, stage, data, config, reference_dir)
    real_step = torch.optim.AdamW.step

    def interrupt(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        signal.raise_signal(signal.SIGINT)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt)
        launch(patch, stage, data, config, interrupted_dir)
    first = torch.load(interrupted_dir / "last.pt", weights_only=True)
    assert first["step"] == 1
    launch(
        monkeypatch,
        stage,
        data,
        config,
        interrupted_dir,
        "--resume",
        str(interrupted_dir / "last.pt"),
    )
    reference = torch.load(reference_dir / "last.pt", weights_only=True)
    resumed = torch.load(interrupted_dir / "last.pt", weights_only=True)
    assert reference["step"] == resumed["step"] == 3
    assert reference["tokens_seen"] == resumed["tokens_seen"]
    assert reference["train_rng"] == resumed["train_rng"]
    assert torch.equal(reference["torch_rng"], resumed["torch_rng"])
    for key, value in reference["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    if stage == "prior":
        source = torch.load(tmp_path / "codec" / "last.pt", weights_only=True)
        for key, value in source["model"].items():
            assert torch.equal(value, resumed["model"]["codec." + key])


def test_low_quality_codec_blocks_prior_before_creating_run(
    stage_files, tmp_path, monkeypatch
):
    data, codec_config, prior_config = stage_files
    launch(monkeypatch, "codec", data, codec_config, tmp_path / "codec")
    config = json.loads(prior_config.read_text())
    config["quality_gate"].update(token_accuracy=0.99, block_accuracy=0.95)
    prior_config.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="尚未达到重建要求"):
        launch(monkeypatch, "prior", data, prior_config, output)
    assert not output.exists()
