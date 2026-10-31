"""验证 token 头条件、因果移位、采样与冻结 VAE。"""

import torch
from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel


def model():
    torch.manual_seed(7)
    return CalmTokenModel(CalmTokenConfig(
        vocab_size=32, chunk_size=4, encoder_dim=8, dim=16,
        layers=1, heads=2, ff_dim=32, decoder_dim=8,
        decoder_layers=1, decoder_heads=2, decoder_ff_dim=16,
        token_head_dim=8, noise_dim=4, energy_layers=1,
        energy_candidates=2, target_samples=2, energy_batch_blocks=4,
    ))


def inputs():
    return (torch.tensor([[1, 4, 5, 6], [1, 7, 8, 0]]), torch.tensor([4, 3]),
            torch.tensor([[9, 10, 11, 12, 13, 14], [15, 16, 17, -100, -100, -100]]))


def test_token_loss_shift_and_frozen_vae_gradients():
    current = model().train()
    prompt, lengths, labels = inputs()
    captured = []
    handle = current.latent_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[1].detach().clone())
    )
    try:
        stats = current.prior_statistics(prompt, lengths, labels)
    finally:
        handle.remove()
    assert captured[0].tolist() == [
        [[1, 9, 10, 11], [1, 13, 14, 0]],
        [[1, 15, 16, 17], [1, 0, 0, 0]],
    ]
    assert stats["tokens"] == stats["objective_count"] == 9
    assert stats["blocks"] == 3
    stats["objective_sum"].backward()
    assert all(p.grad is None and not p.requires_grad for p in current.codec.parameters())
    assert not current.codec.training
    for component in (current.history_compressor, current.backbone, current.latent_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())


def test_sample_is_reproducible_and_encodes_gru_tokens():
    current = model().eval()
    context = torch.randn(2, current.cfg.dim)
    noise = current.draw_noise((2,), "cpu", torch.Generator().manual_seed(42))
    captured = []
    original = current.codec.encode

    def record(ids, lengths):
        captured.append((ids.detach().clone(), lengths.detach().clone()))
        return original(ids, lengths)

    current.codec.encode = record
    try:
        a = current.sample_latent(context, noise=noise)
        b = current.sample_latent(context, noise=noise)
    finally:
        current.codec.encode = original
    torch.testing.assert_close(a, b)
    assert captured[0][1].tolist() == [4, 4]
    assert not torch.isin(captured[0][0], torch.tensor([0, 1])).any()
    expected, _ = original(*captured[0])
    torch.testing.assert_close(a, expected[:, 0])


def test_generated_text_and_causal_head_logits():
    current = model().eval()
    prompt, lengths, _ = inputs()
    generated = current.generate(prompt, lengths, max_new_tokens=4, min_new_tokens=4,
                                 generator=torch.Generator().manual_seed(99), return_trace=True)
    assert generated["output_ids"].shape == (2, 4)
    h = torch.randn(2, current.cfg.dim)
    previous = torch.tensor([[1, 4, 5, 6], [1, 5, 6, 7]])
    changed = previous.clone()
    changed[:, 2:] = torch.tensor([12, 13])
    old = current.latent_head(h, previous)
    new = current.latent_head(h, changed)
    torch.testing.assert_close(old[:, :2], new[:, :2])
    assert not torch.allclose(old[:, 2:], new[:, 2:])


def test_block_gru_does_not_carry_state_between_blocks():
    current = model().eval()
    contexts = torch.randn(2, 3, current.cfg.dim)
    previous = torch.tensor([
        [[1, 4, 5, 6], [1, 7, 8, 9], [1, 10, 11, 12]],
        [[1, 5, 6, 7], [1, 8, 9, 10], [1, 11, 12, 13]],
    ])
    changed = previous.clone()
    changed[:, 0, 1:] = torch.tensor([19, 20, 21])
    with torch.no_grad():
        joint = current.latent_head(contexts, previous)
        after_change = current.latent_head(contexts, changed)
        independently = torch.stack([
            torch.stack([current.latent_head(contexts[b, t], previous[b, t])
                         for t in range(contexts.shape[1])])
            for b in range(contexts.shape[0])
        ])
    torch.testing.assert_close(joint[:, 1:], after_change[:, 1:])
    torch.testing.assert_close(joint, independently, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(joint[:, 0], after_change[:, 0])


def test_sampling_does_not_keep_gru_hidden_state_between_calls():
    current = model().eval()
    context = torch.randn(2, current.cfg.dim)
    noise = current.draw_noise((2,), "cpu", torch.Generator().manual_seed(100))
    with torch.no_grad():
        first = current.sample_latent(context, noise=noise)
        current.sample_latent(torch.randn_like(context), noise=1 - noise)
        repeated = current.sample_latent(context, noise=noise)
    torch.testing.assert_close(first, repeated)
