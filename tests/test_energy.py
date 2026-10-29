"""Energy 分布目标、随机生成和纯 latent 回灌的行为验证。"""

from dataclasses import asdict
import math

import pytest
import torch

from latent_ar.checkpoints import load_model
from latent_ar.energy import (
    EnergyHead,
    EnergyModelConfig,
    LatentEnergyModel,
    energy_statistics,
    decoder_mixture_statistics,
    decoder_target_log_probs,
)
from latent_ar.staged_training import initialize_backbone, evaluate_stage
from test_staged import examples, small_config


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    return LatentEnergyModel(
        EnergyModelConfig(
            **asdict(small_config()),
            noise_dim=8,
            energy_hidden_dim=32,
            energy_layers=2,
            energy_candidates=4,
        )
    )


def test_energy_matches_off_diagonal_euclidean_estimator():
    samples = torch.tensor([[[[-2.0, 0.0], [1.0, 1.0], [3.0, -1.0]]]])
    target = torch.tensor([[[0.5, 0.5]]])
    stats = energy_statistics(samples, target, torch.ones(1, 1, dtype=torch.bool))
    candidates = samples[0, 0]
    attraction = sum(torch.linalg.vector_norm(z - target[0, 0]) for z in candidates) / 3
    spread = (
        sum(
            torch.linalg.vector_norm(a - b)
            for i, a in enumerate(candidates)
            for j, b in enumerate(candidates)
            if i != j
        )
        / 6
    )
    torch.testing.assert_close(
        stats["energy_sum"], (attraction - 0.5 * spread) / math.sqrt(2)
    )
    torch.testing.assert_close(stats["spread_sum"], spread / math.sqrt(2))


def test_energy_prefers_true_two_mode_distribution_to_its_average():
    # 穷举两个独立 Bernoulli 候选及目标，避免用随机误差掩盖评分性质。
    scores = []
    collapsed = []
    valid = torch.ones(1, 1, dtype=torch.bool)
    for target in [-1.0, 1.0]:
        truth = torch.tensor([[[target]]])
        for a in [-1.0, 1.0]:
            for b in [-1.0, 1.0]:
                samples = torch.tensor([[[[a], [b]]]])
                scores.append(energy_statistics(samples, truth, valid)["energy_sum"])
                collapsed.append(
                    energy_statistics(torch.zeros_like(samples), truth, valid)[
                        "energy_sum"
                    ]
                )
    assert torch.stack(scores).mean() == 0.5
    assert torch.stack(collapsed).mean() == 1.0


def test_generator_learns_two_modes_instead_of_their_mean():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    cfg = EnergyModelConfig(
        vocab_size=32,
        dim=8,
        heads=2,
        layers=1,
        ff_dim=16,
        encoder_dim=8,
        decoder_dim=8,
        decoder_heads=2,
        decoder_ff_dim=16,
        noise_dim=4,
        energy_hidden_dim=32,
        energy_layers=2,
    )
    head = EnergyHead(cfg)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0)
    context = torch.zeros(64, 1, 1, 8)
    valid = torch.ones(64, 1, dtype=torch.bool)
    for _ in range(600):
        target = (torch.randint(0, 2, (64, 1, 1)).float() * 2 - 1).expand(-1, -1, 8)
        samples = head(context, torch.randn(64, 1, 8, 4))
        loss = energy_statistics(samples, target, valid)["energy_sum"] / 64
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        samples = head(torch.zeros(4096, 8), torch.randn(4096, 4))
        positive_fraction = (samples.mean(-1) > 0).float().mean()
        nearest_mode = (
            torch.minimum(
                (samples - 1).square().mean(-1), (samples + 1).square().mean(-1)
            )
            .sqrt()
            .mean()
        )
    assert 0.4 < positive_fraction < 0.6
    assert nearest_mode < 0.1


def test_padding_is_excluded_and_identical_candidates_have_finite_gradients():
    samples = torch.ones(2, 3, 4, 8, requires_grad=True)
    targets = torch.zeros(2, 3, 8)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    stats = energy_statistics(samples, targets, valid)
    stats["energy_sum"].backward()
    assert stats["blocks"] == 3
    assert torch.isfinite(samples.grad).all()
    assert torch.count_nonzero(samples.grad[~valid]) == 0
    assert samples.grad[valid].abs().sum() > 0


def test_energy_rejects_one_candidate():
    with pytest.raises(ValueError):
        energy_statistics(
            torch.zeros(1, 1, 1, 4),
            torch.zeros(1, 1, 4),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_energy_gradients_reach_backbone_and_noise_while_codec_stays_frozen(model):
    model.train()
    stats = model.prior_statistics(
        *examples(), generator=torch.Generator().manual_seed(42)
    )
    assert not stats["sample_ce_sum"].requires_grad
    stats["energy_sum"].backward()
    assert not model.codec.training
    assert all(p.grad is None and not p.requires_grad for p in model.codec.parameters())
    for module in [model.backbone, model.latent_head.input, model.latent_head.output]:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_sampling_is_seeded_but_not_deterministic_across_seeds(model):
    prompt, lengths, _ = examples()
    first = model.rollout(
        prompt, lengths, 3, generator=torch.Generator().manual_seed(42)
    )
    second = model.rollout(
        prompt, lengths, 3, generator=torch.Generator().manual_seed(42)
    )
    other = model.rollout(
        prompt, lengths, 3, generator=torch.Generator().manual_seed(43)
    )
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, other)


def test_first_prediction_matches_teacher_with_shared_noise_and_feedback_is_latent(
    model,
):
    prompt, lengths, labels = examples()
    contexts, _, _ = model.teacher_contexts(prompt, lengths, labels)
    noise = model.draw_noise(
        (2, 3), torch.device("cpu"), torch.Generator().manual_seed(42)
    )
    expected = model.sample_latent(contexts[:, 0], noise=noise[:, 0])
    seen = []
    hook = model.backbone.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone())
    )
    actual = model.rollout(prompt, lengths, 3, noise=noise)
    hook.remove()
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(seen[1][:, 0], actual[:, 0])
    torch.testing.assert_close(seen[2][:, 0], actual[:, 1])


def test_token_sampling_does_not_change_latents(model, monkeypatch):
    prompt, lengths, _ = examples()
    results = []
    rollout = model.rollout

    def capture(*args, **kwargs):
        value = rollout(*args, **kwargs)
        results.append(value.clone())
        return value

    monkeypatch.setattr(model, "rollout", capture)
    for temperature in [0.0, 0.8]:
        model.generate(
            prompt,
            lengths,
            max_new_tokens=8,
            temperature=temperature,
            generator=torch.Generator().manual_seed(42),
        )
    torch.testing.assert_close(results[0], results[1])


def test_energy_checkpoint_preserves_weights_and_seeded_generation(model):
    checkpoint = dict(
        architecture=model.architecture,
        model_config=asdict(model.cfg),
        model=model.state_dict(),
    )
    loaded = load_model(checkpoint, torch.device("cpu"))
    prompt, lengths, _ = examples()
    a = model.rollout(prompt, lengths, 3, generator=torch.Generator().manual_seed(5))
    b = loaded.rollout(prompt, lengths, 3, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(a, b)


def test_mixture_averages_probabilities_and_keeps_one_candidate_for_whole_block():
    # 两候选分别偏向不同位置；整块概率须为 mean(.9*.1, .1*.9)=.09，
    # 不能把每位置边际概率相乘成 .5*.5=.25。第三个 padding 位置忽略。
    probabilities = torch.tensor([[[0.9, 0.1], [0.1, 0.9], [0.01, 0.99]]])
    valid = torch.tensor([[True, True, False]])
    stats = decoder_mixture_statistics(probabilities.log(), valid, 2)
    torch.testing.assert_close(
        stats["mixture_token_ce_sum"], torch.tensor(-2 * math.log(0.5))
    )
    torch.testing.assert_close(
        stats["mixture_block_nll_sum"], torch.tensor(-math.log(0.09))
    )


def test_distribution_diagnostics_do_not_change_energy_or_rng_samples(model):
    options = examples()
    base = model.prior_statistics(*options, generator=torch.Generator().manual_seed(42))
    diagnostic = model.prior_statistics(
        *options,
        generator=torch.Generator().manual_seed(42),
        distribution_diagnostics=True,
    )
    torch.testing.assert_close(base["energy_sum"], diagnostic["energy_sum"])
    torch.testing.assert_close(base["sample_ce_sum"], diagnostic["sample_ce_sum"])
    assert torch.isfinite(diagnostic["mixture_block_nll_sum"])
    assert not diagnostic["mixture_block_nll_sum"].requires_grad


@pytest.mark.parametrize("objective", ["token_ce", "block_mixture"])
def test_text_objectives_train_through_frozen_decoder(model, objective):
    model.cfg.objective = objective
    stats = model.prior_statistics(
        *examples(), generator=torch.Generator().manual_seed(42)
    )
    assert stats["objective_count"] == stats["tokens"]
    stats["objective_sum"].backward()
    assert model.latent_head.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.codec.parameters())


def test_cycle_uses_complete_blocks_and_does_not_train_codec(model):
    model.cfg.objective = "block_mixture"
    model.cfg.cycle_weight = 1.0
    stats = model.prior_statistics(
        *examples(), generator=torch.Generator().manual_seed(42)
    )
    assert stats["cycle_blocks"] == 5
    loss = (
        stats["objective_sum"] / stats["tokens"]
        + stats["weighted_cycle_sum"] / stats["cycle_blocks"]
    )
    loss.backward()
    assert model.latent_head.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.codec.parameters())


def test_checkpointed_decoder_gradient_matches_direct_probability_gradient(model):
    _, _, labels = examples()
    candidates = torch.randn(2, 3, 3, model.cfg.dim, requires_grad=True)
    actual = decoder_target_log_probs(model, candidates, labels)
    actual_loss = decoder_mixture_statistics(actual, labels != -100, 4)[
        "mixture_block_nll_sum"
    ]
    actual_grad = torch.autograd.grad(actual_loss, candidates)[0]
    direct_candidates = candidates.detach().clone().requires_grad_(True)
    direct = []
    for index in range(3):
        logits = model.decode(
            model.denormalize(direct_candidates[:, :, index]), labels.shape[1]
        )
        direct.append(
            logits.float()
            .log_softmax(-1)
            .gather(-1, labels.clamp_min(0)[..., None])
            .squeeze(-1)
        )
    expected_loss = decoder_mixture_statistics(
        torch.stack(direct, -1), labels != -100, 4
    )["mixture_block_nll_sum"]
    expected_grad = torch.autograd.grad(expected_loss, direct_candidates)[0]
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_backbone_initialization_requires_identical_codec(model, tmp_path):
    from latent_ar.staged import LatentPriorModel

    source = LatentPriorModel(small_config())
    source.codec.load_state_dict(model.codec.state_dict())
    source.set_latent_statistics(torch.randn(16, source.cfg.dim))
    path = tmp_path / "source.pt"
    checkpoint = dict(
        architecture=source.architecture,
        model=source.state_dict(),
        dataset_hash="same",
        step=12,
    )
    torch.save(checkpoint, path)
    initialize_backbone(model, path, "same")
    for key, value in source.backbone.state_dict().items():
        torch.testing.assert_close(model.backbone.state_dict()[key], value)
    torch.testing.assert_close(model.latent_std, source.latent_std)
    checkpoint["model"]["codec.decoder.expand.weight"] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="same frozen codec"):
        initialize_backbone(model, path, "same")
