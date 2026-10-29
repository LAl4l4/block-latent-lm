from dataclasses import asdict

import torch

from latent_ar.checkpoints import load_model
from latent_ar.denoising import DenoisingConfig, LatentDenoisingModel
from test_staged import examples, small_config


def make_model():
    return LatentDenoisingModel(
        DenoisingConfig(
            **asdict(small_config()),
            denoise_hidden_dim=64,
            denoise_layers=2,
            denoise_steps=32,
            context_dropout=0.0,
            denoise_guidance=1.0
        )
    )


def test_denoising_uses_fixed_codec_targets_and_gradients_reach_backbone():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = make_model()
    stats = model.prior_statistics(*examples(), token_ce_grad=False)
    assert not stats["token_ce_sum"].requires_grad
    stats["latent_mse_sum"].backward()
    assert model.latent_head.context.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    assert all(p.grad is None for p in model.codec.parameters())
    prompt, lengths, labels = examples()
    model.eval()
    context, targets, _ = model.teacher_contexts(prompt, lengths, labels)
    noise = model.draw_noise(
        targets.shape[:2], prompt.device, torch.Generator().manual_seed(1)
    )
    teacher = model.sample_latent(context, noise=noise)
    free = model.rollout(prompt, lengths, targets.shape[1], noise=noise)
    torch.testing.assert_close(teacher[:, 0], free[:, 0])
    restored = load_model(
        {
            "architecture": model.architecture,
            "model_config": asdict(model.cfg),
            "model": model.state_dict(),
        },
        prompt.device,
    )
    torch.testing.assert_close(teacher, restored.sample_latent(context, noise=noise))


def test_clean_latent_regression_can_generate_two_modes_without_energy():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = make_model()
    optimizer = torch.optim.Adam(model.latent_head.parameters(), lr=0.002)
    context = torch.zeros(128, model.cfg.dim)
    for _ in range(800):
        target = (
            (torch.randint(0, 2, (128, 1)) * 2 - 1).float().expand(-1, model.cfg.dim)
        )
        noise = torch.randn_like(target)
        time = torch.rand(128, 1)
        prediction = model.latent_head(
            (1 - time) * noise + time * target, time, context
        )
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = model.sample_latent(torch.zeros(2048, model.cfg.dim))
    nearest = torch.minimum(
        (output - 1).square().mean(-1), (output + 1).square().mean(-1)
    ).sqrt()
    assert nearest.mean() < 0.15
    fraction = (output.mean(-1) > 0).float().mean()
    assert 0.3 < fraction < 0.7
