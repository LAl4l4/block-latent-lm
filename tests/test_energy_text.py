from dataclasses import asdict

import torch

from latent_ar.checkpoints import load_model
from latent_ar.energy_text import (
    EnergyTextConfig,
    LatentEnergyTextModel,
    decoder_brier_loss,
)
from test_staged import examples, small_config


def test_brier_matches_explicit_probability_products_and_padding_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    labels = torch.tensor([[1, 3, 2, 0], [4, 2, -100, -100]])
    p = logits.softmax(-1)
    expected = []
    for row in range(2):
        loss = 0.0
        for n in range(1, 5):
            terms = []
            for start in range(5 - n):
                truth = labels[row, start : start + n]
                if bool((truth == -100).any()):
                    continue
                hit = (
                    sum(
                        torch.stack(
                            [p[row, i, start + s, truth[s]] for s in range(n)]
                        ).prod()
                        for i in range(3)
                    )
                    / 3
                )
                collision = (
                    sum(
                        torch.stack(
                            [
                                (p[row, i, start + s] * p[row, j, start + s]).sum()
                                for s in range(n)
                            ]
                        ).prod()
                        for i in range(3)
                        for j in range(3)
                        if i != j
                    )
                    / 6
                )
                terms.append(collision - 2 * hit)
            if terms:
                loss = loss + torch.stack(terms).mean()
        expected.append(loss)
    actual = decoder_brier_loss(logits, labels)
    torch.testing.assert_close(actual, torch.stack(expected))
    actual.sum().backward()
    assert logits.grad[1, :, 2:].abs().max() == 0


def test_text_score_gradients_freeze_codec_and_preserve_checkpoint():
    torch.manual_seed(4)
    model = LatentEnergyTextModel(
        EnergyTextConfig(
            **asdict(small_config()),
            noise_dim=8,
            energy_hidden_dim=32,
            energy_layers=1,
            energy_candidates=3,
            text_score_blocks=2
        )
    )
    stats = model.prior_statistics(*examples())
    # 单独文字损失也必须给主干和随机头梯度。
    stats["text_brier_sum"].backward()
    assert model.latent_head.input.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    assert all(p.grad is None for p in model.codec.parameters())
    checkpoint = {
        "architecture": model.architecture,
        "model_config": asdict(model.cfg),
        "model": model.state_dict(),
    }
    restored = load_model(checkpoint, torch.device("cpu"))
    prompt, lengths, _ = examples()
    left = model.rollout(
        prompt, lengths, 2, generator=torch.Generator().manual_seed(42)
    )
    right = restored.rollout(
        prompt, lengths, 2, generator=torch.Generator().manual_seed(42)
    )
    torch.testing.assert_close(left, right)
