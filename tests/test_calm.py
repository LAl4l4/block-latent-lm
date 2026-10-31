import torch

from latent_ar.calm import CalmConfig, CalmModel, posterior_energy_statistics


def tiny():
    return CalmConfig(
        vocab_size=32,
        chunk_size=2,
        encoder_dim=8,
        dim=8,
        layers=1,
        heads=2,
        ff_dim=16,
        decoder_dim=8,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ff_dim=16,
        noise_dim=3,
        energy_layers=1,
        energy_candidates=3,
        target_samples=5,
        energy_batch_blocks=1,
    )


def inputs():
    return (
        torch.tensor([[1, 4, 5], [1, 7, 0]]),
        torch.tensor([3, 2]),
        torch.tensor([[6, 8, 9, 10], [11, 12, 13, -100]]),
    )


def test_energy_matches_direct_ordered_pairs():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 4, requires_grad=True)
    y = torch.randn(2, 5, 4)
    score, attraction, spread = posterior_energy_statistics(x, y)
    expected_a = (x[:, :, None] - y[:, None, :]).norm(dim=-1).mean((1, 2))
    expected_s = (
        sum(
            (x[:, i] - x[:, j]).norm(dim=-1)
            for i in range(3)
            for j in range(3)
            if i != j
        )
        / 6
    )
    torch.testing.assert_close(attraction, expected_a.sum())
    torch.testing.assert_close(spread, expected_s.sum())
    torch.testing.assert_close(score, (2 * expected_a - expected_s).sum())
    score.backward()
    assert torch.isfinite(x.grad).all()


def test_shuffled_context_never_uses_invalid_history():
    from latent_ar.calm import shuffle_valid_contexts

    contexts = torch.tensor([[[1.], [10.]], [[2.], [999.]], [[3.], [30.]]])
    valid = torch.tensor([[True, True], [True, False], [True, True]])
    shuffled = shuffle_valid_contexts(contexts, valid)
    assert shuffled[:, 0, 0].tolist() == [3., 1., 2.]
    assert shuffled[[0, 2], 1, 0].tolist() == [30., 10.]
    assert shuffled[1, 1, 0] == 999.


def test_future_labels_do_not_leak_into_context():
    torch.manual_seed(5)
    model = CalmModel(tiny()).eval()
    prompt, lengths, labels = inputs()
    updated = labels.clone()
    updated[0, 0:2] = torch.tensor([14, 15])
    old_h, _, _, _ = model.teacher_contexts(prompt, lengths, labels)
    new_h, _, _, _ = model.teacher_contexts(prompt, lengths, updated)
    torch.testing.assert_close(old_h[:, 0], new_h[:, 0])
    assert not torch.allclose(old_h[0, 1], new_h[0, 1])


def test_energy_gradients_freeze_vae_and_cover_blocks():
    torch.manual_seed(6)
    model = CalmModel(tiny()).train()
    # 官方的末层零初始化使第一步的上游梯度为零。
    torch.nn.init.normal_(model.latent_head.output.weight, std=0.1)
    prompt, lengths, labels = inputs()
    stats = model.prior_statistics(
        prompt, lengths, labels, generator=torch.Generator().manual_seed(8)
    )
    assert stats["blocks"] == 4 and stats["tokens"] == 7
    stats["energy_sum"].backward()
    assert all(p.grad is None for p in model.codec.parameters())
    for module in (model.history_compressor, model.backbone, model.latent_head):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert not model.codec.training


def test_sampling_reproducible_and_uniform_noise():
    model = CalmModel(tiny())
    a = model.draw_noise((100,), "cpu", torch.Generator().manual_seed(7))
    b = model.draw_noise((100,), "cpu", torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert a.min() >= -0.5 and a.max() < 0.5


def test_zero_initialization_gradients_reach_backbone_after_one_step():
    torch.manual_seed(11)
    model = CalmModel(tiny()).train()
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=0.05
    )
    args = inputs()
    first = model.prior_statistics(*args, generator=torch.Generator().manual_seed(1))
    first["energy_sum"].backward()
    assert model.latent_head.output.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None or p.grad.abs().sum() == 0 for p in model.backbone.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model.prior_statistics(*args, generator=torch.Generator().manual_seed(2))
    second["energy_sum"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )


def test_checkpoint_matches_eager_all_blocks(monkeypatch):
    import latent_ar.calm as calm

    torch.manual_seed(12)
    model = CalmModel(tiny()).train()
    torch.nn.init.normal_(model.latent_head.output.weight, std=0.1)
    args = inputs()
    checkpointed = model.prior_statistics(
        *args, generator=torch.Generator().manual_seed(4)
    )
    checkpointed["energy_sum"].backward()
    grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(calm, "checkpoint", lambda f, *a, **kw: f(*a))
    eager = model.prior_statistics(*args, generator=torch.Generator().manual_seed(4))
    eager["energy_sum"].backward()
    assert eager["blocks"] == checkpointed["blocks"] == 4
    torch.testing.assert_close(eager["energy_sum"], checkpointed["energy_sum"])
    for name, parameter in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(parameter.grad, grads[name])
