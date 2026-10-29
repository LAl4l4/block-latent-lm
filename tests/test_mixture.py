"""联合文字分布、冻结梯度、随机分量和旧检查点隔离。"""

from dataclasses import asdict
import math

import pytest
import torch

from latent_ar.checkpoints import load_model
from latent_ar.mixture import LatentMixtureModel, MixtureModelConfig, mixture_block_nll
from test_staged import examples, small_config
from evaluate_text_distribution import brier_ngrams


def test_joint_nll_matches_enumerated_distribution_and_marginalizes_padding():
    torch.manual_seed(12)
    logits = torch.randn(1, 3, 2, 5, requires_grad=True)
    weights = torch.randn(1, 3, requires_grad=True)
    labels = torch.tensor([[2, 4]])
    probability = logits.softmax(-1)
    expected = (
        -(weights.softmax(-1) * probability[:, :, 0, 2] * probability[:, :, 1, 4])
        .sum()
        .log()
    )
    actual = mixture_block_nll(logits, weights, labels)
    torch.testing.assert_close(actual[0], expected)
    # 对所有可能完整块求和必须等于 1，检查求和顺序和归一化。
    total = sum(
        mixture_block_nll(logits, weights, torch.tensor([[a, b]])).neg().exp()
        for a in range(5)
        for b in range(5)
    )
    torch.testing.assert_close(total, torch.ones_like(total))
    partial = mixture_block_nll(logits, weights, torch.tensor([[2, -100]]))
    torch.testing.assert_close(
        partial[0], -(weights.softmax(-1) * probability[:, :, 0, 2]).sum().log()
    )
    partial.sum().backward()
    assert logits.grad[:, :, 1].abs().max() == 0
    assert weights.grad.abs().sum() > 0


def test_joint_nll_does_not_choose_a_different_component_for_each_slot():
    # 分量分别只能生成 00 / 11；01 必须很不可能。
    logits = torch.tensor(
        [[[[20.0, -20.0], [20.0, -20.0]], [[-20.0, 20.0], [-20.0, 20.0]]]]
    )
    loss = mixture_block_nll(logits, torch.zeros(1, 2), torch.tensor([[0, 1]]))
    assert loss.item() > 30


@pytest.fixture
def mixture():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    return LatentMixtureModel(
        MixtureModelConfig(
            **asdict(small_config()),
            mixture_components=3,
            mixture_hidden_dim=32,
            mixture_layers=1,
            mixture_train_blocks=2
        )
    )


def test_text_nll_reaches_head_and_backbone_but_freezes_codec(mixture):
    batch = dict(zip(("prompt", "prompt_lengths", "labels"), examples()))
    stats = mixture.prior_statistics(
        batch["prompt"], batch["prompt_lengths"], batch["labels"]
    )
    (stats["text_nll_sum"] / stats["tokens"]).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in mixture.backbone.parameters()
    )
    assert mixture.latent_head.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in mixture.codec.parameters())
    assert stats["scored_blocks"] <= 2
    mixture.eval()
    stats = mixture.prior_statistics(
        batch["prompt"], batch["prompt_lengths"], batch["labels"]
    )
    assert stats["scored_blocks"] == stats["blocks"]


def test_sampling_cache_first_block_and_checkpoint(mixture):
    batch = dict(zip(("prompt", "prompt_lengths", "labels"), examples()))
    mixture.eval()
    contexts, targets, _ = mixture.teacher_contexts(
        batch["prompt"], batch["prompt_lengths"], batch["labels"]
    )
    noise = torch.rand((*targets.shape[:2], 1))
    teacher = mixture.sample_latent(contexts, noise=noise)
    free = mixture.rollout(
        batch["prompt"], batch["prompt_lengths"], targets.shape[1], noise=noise
    )
    torch.testing.assert_close(teacher[:, 0], free[:, 0])
    checkpoint = {
        "architecture": mixture.architecture,
        "model_config": asdict(mixture.cfg),
        "model": mixture.state_dict(),
    }
    restored = load_model(checkpoint, torch.device("cpu"))
    torch.testing.assert_close(teacher, restored.sample_latent(contexts, noise=noise))


def test_two_modes_are_fitted_without_forcing_each_component_to_both_targets():
    # 与平均逐候选 CE 的退化行为区分：每个分量可以专注一个模式。
    torch.set_num_threads(1)
    torch.manual_seed(7)
    logits = torch.nn.Parameter(torch.randn(1, 2, 2, 2))
    weights = torch.nn.Parameter(torch.zeros(1, 2))
    optimizer = torch.optim.Adam([logits, weights], lr=0.1)
    for _ in range(150):
        loss = (
            sum(
                mixture_block_nll(logits, weights, torch.tensor([[y, y]])).sum()
                for y in [0, 1]
            )
            / 2
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < math.log(2) + 0.03
    assert mixture_block_nll(logits, weights, torch.tensor([[0, 1]])).item() > 5


def test_brier_uses_independent_pairs_and_ignores_incomplete_ngrams():
    samples = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    labels = torch.tensor([[1, 2, 3, -100]])
    score = brier_ngrams(samples, labels)
    for n in ["1", "2", "3"]:
        assert score[n]["reward"] == 1.0
    assert score["4"]["reward"] is None
