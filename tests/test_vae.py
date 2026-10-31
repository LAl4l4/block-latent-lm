"""块级 VAE 的采样、掩码、自由位与梯度检查。"""

import torch

from latent_ar.vae import BlockVAE, VAEConfig


def make_model(**kwargs):
    torch.manual_seed(8)
    cfg = VAEConfig(
        vocab_size=32,
        chunk_size=4,
        encoder_dim=8,
        dim=16,
        layers=1,
        heads=2,
        ff_dim=32,
        decoder_dim=8,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ff_dim=16,
        **kwargs,
    )
    return BlockVAE(cfg)


def test_posterior_shapes_padding_and_no_posterior_rmsnorm():
    model = make_model().eval()
    ids = torch.tensor([[3, 4, 5, 6, 7, 0, 0, 0], [3, 4, 5, 6, 7, 8, 9, 10]])
    lengths = torch.tensor([5, 8])
    mean, log_std, valid = model.encode_posterior(ids, lengths)
    assert mean.shape == log_std.shape == (2, 2, 16)
    assert valid.tolist() == [[True, True], [True, True]]
    assert model.decoder.output.weight is model.encoder.embedding.weight
    with torch.no_grad():
        model.posterior.weight.zero_()
        model.posterior.bias[:16].fill_(2.0)
        model.posterior.bias[16:].fill_(9.0)
    mean, log_std, _ = model.encode_posterior(ids, lengths)
    assert torch.all(mean == 2.0) and torch.all(log_std == 4.0)


def test_loss_counts_partial_blocks_kl_mask_and_gradients():
    model = make_model(vae_kl_floor=100.0)
    ids = torch.tensor([[3, 4, 5, 6, 7, 0, 0, 0], [3, 4, 5, 6, 0, 0, 0, 0]])
    lengths = torch.tensor([5, 4])
    labels = ids.masked_fill(torch.arange(8)[None, :] >= lengths[:, None], -100)
    stats = model.reconstruction_statistics(
        ids, lengths, labels, torch.Generator().manual_seed(12)
    )
    assert stats["tokens"] == 9 and stats["blocks"] == 3 and stats["full_blocks"] == 2
    torch.testing.assert_close(stats["kl_sum"], torch.tensor(4800.0))
    torch.testing.assert_close(
        stats["loss_sum"], stats["ce_sum"] + 0.001 * stats["kl_sum"]
    )
    stats["loss_sum"].backward()
    assert model.posterior.weight.grad is not None
    assert torch.isfinite(model.posterior.weight.grad).all()
    assert model.encoder.mlp[0].weight.grad.abs().sum() > 0


def test_eval_samples_are_seeded_and_corruption_disabled():
    model = make_model(vae_input_dropout=0.9, vae_latent_dropout=0.9).eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    lengths = torch.tensor([4])
    original = model.encode_posterior(ids, lengths, corrupt=False)
    uncorrupted = model.encode_posterior(ids, lengths, corrupt=True)
    for a, b in zip(original, uncorrupted):
        torch.testing.assert_close(a, b)
    first = model(ids, lengths, torch.Generator().manual_seed(4))
    same = model(ids, lengths, torch.Generator().manual_seed(4))
    other = model(ids, lengths, torch.Generator().manual_seed(5))
    torch.testing.assert_close(first, same)
    assert not torch.allclose(first, other)
