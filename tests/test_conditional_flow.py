import torch

from latent_ar.conditional_flow import ConditionalFlowModel, FlowConfig, flow_interpolate


def make_model():
    cfg = FlowConfig(vocab_size=32, chunk_size=2, encoder_dim=8, dim=8, layers=1,
                     heads=2, ff_dim=16, decoder_dim=8, decoder_layers=1,
                     decoder_heads=2, decoder_ff_dim=16, noise_dim=3,
                     energy_layers=1, energy_candidates=2, target_samples=2,
                     energy_batch_blocks=2, flow_steps=3)
    return ConditionalFlowModel(cfg)


def inputs():
    return (torch.tensor([[1, 4, 5], [1, 7, 0]]), torch.tensor([3, 2]),
            torch.tensor([[6, 8, 9, 10], [11, 12, 13, -100]]))


def test_linear_path_and_velocity():
    x = torch.tensor([[1., 2.]])
    y = torch.tensor([[5., 10.]])
    z, velocity = flow_interpolate(x, y, torch.tensor([[0.25]]))
    torch.testing.assert_close(z, torch.tensor([[2., 4.]]))
    torch.testing.assert_close(velocity, y - x)
    torch.testing.assert_close(flow_interpolate(x, y, torch.zeros(1, 1))[0], x)
    torch.testing.assert_close(flow_interpolate(x, y, torch.ones(1, 1))[0], y)


def test_training_gradient_and_frozen_vae():
    torch.manual_seed(10)
    model = make_model().train()
    torch.nn.init.normal_(model.latent_head.output[-1].weight, std=0.1)
    values = model.prior_statistics(*inputs(), generator=torch.Generator().manual_seed(4))
    assert values["blocks"] == 4 and values["tokens"] == 7
    values["flow_sum"].backward()
    for part in (model.backbone, model.history_compressor, model.latent_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert all(p.grad is None for p in model.codec.parameters())
    assert not model.codec.training


def test_first_context_does_not_read_current_target():
    model = make_model().eval()
    prompt, lengths, labels = inputs()
    changed = labels.clone()
    changed[0, :2] = torch.tensor([17, 18])
    h1 = model.teacher_contexts(prompt, lengths, labels)[0]
    h2 = model.teacher_contexts(prompt, lengths, changed)[0]
    torch.testing.assert_close(h1[:, 0], h2[:, 0])


def test_euler_sampling_is_reproducible_and_honors_explicit_noise():
    model = make_model().eval()
    torch.nn.init.normal_(model.latent_head.output[-1].weight, std=0.1)
    h = torch.randn(2, model.cfg.dim)
    a = model.sample_latent(h, generator=torch.Generator().manual_seed(8), steps=4)
    b = model.sample_latent(h, generator=torch.Generator().manual_seed(8), steps=4)
    torch.testing.assert_close(a, b)
    noise = torch.randn_like(h)
    torch.testing.assert_close(model.sample_latent(h, noise=noise, steps=4),
                               model.sample_latent(h, noise=noise, steps=4))
    assert not torch.equal(model.sample_latent(h, noise=noise, steps=1),
                           model.sample_latent(h, noise=noise, steps=4))
