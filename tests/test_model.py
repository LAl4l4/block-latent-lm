"""V2 核心约束：块内双向、块间隔离、只读 latent、训练可微。"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from latent_ar.data import collate
from latent_ar.model import LatentARModel, LocalSelfAttention, ModelConfig


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return LatentARModel(
        ModelConfig(
            vocab_size=32,
            encoder_dim=8,
            dim=16,
            layers=2,
            heads=2,
            ff_dim=32,
            decoder_dim=16,
            decoder_layers=1,
            decoder_heads=2,
            decoder_ff_dim=32,
        )
    )


def inputs():
    return (
        torch.tensor([[4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 0, 0, 0, 0]]),
        torch.tensor([7, 3]),
    )


def test_local_attention_is_bidirectional():
    """扰动第四个位置，第一个位置必须能随之变化；因果 attention 做不到。"""
    attention = LocalSelfAttention(8, 2)
    with torch.no_grad():
        attention.qkv.weight.zero_()
        # Q/K 为零使四个位置均匀互看；V 与输出均为恒等映射。
        attention.qkv.weight[16:].copy_(torch.eye(8))
        attention.out.weight.copy_(torch.eye(8))
    hidden = torch.zeros(1, 4, 8)
    baseline = attention(hidden)
    hidden[:, 3] = 4
    changed = attention(hidden)
    torch.testing.assert_close(baseline[:, 0], torch.zeros(1, 8))
    torch.testing.assert_close(changed[:, 0], torch.ones(1, 8))


def test_decoder_blocks_are_isolated(model):
    latents = torch.randn(2, 3, model.cfg.dim, requires_grad=True)
    baseline = model.decoder(latents)
    changed_latents = latents.detach().clone()
    changed_latents[:, 1] += torch.randn_like(changed_latents[:, 1])
    changed = model.decoder(changed_latents)
    torch.testing.assert_close(baseline[:, 0], changed[:, 0])
    torch.testing.assert_close(baseline[:, 2], changed[:, 2])
    assert not torch.allclose(baseline[:, 1], changed[:, 1])
    baseline[:, 0].square().mean().backward()
    assert torch.count_nonzero(latents.grad[:, 1:]) == 0
    assert latents.grad[:, 0].abs().sum() > 0


def test_no_text_input_or_prompt_bypass(model, monkeypatch):
    assert not hasattr(model.decoder, "embedding")
    assert not hasattr(model.decoder.blocks[0], "cross")
    assert not hasattr(model.decoder, "project_memory")
    prompt, lengths = inputs()
    fixed = torch.randn(2, 2, model.cfg.dim)
    monkeypatch.setattr(model, "rollout", lambda *args, **kwargs: fixed)
    first = model(prompt, lengths, 8)
    second = model(torch.full_like(prompt, 20), lengths, 8)
    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="output_tokens"):
        model(prompt, lengths, torch.ones(2, 8, dtype=torch.long))


def test_padding_does_not_change_rollout(model):
    prompt, lengths = inputs()
    first = model.rollout(prompt, lengths, 3)
    second = model.rollout(F.pad(prompt, (0, 6), value=31), lengths, 3)
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    for i, length in enumerate(lengths.tolist()):
        alone = model.rollout(prompt[i : i + 1, :length], lengths[i : i + 1], 3)
        torch.testing.assert_close(first[i], alone[0], atol=2e-6, rtol=2e-5)


def test_later_latents_cannot_change_earlier_blocks(model):
    prompt, lengths = inputs()
    first = model.rollout(prompt, lengths, 2)
    longer = model.rollout(prompt, lengths, 4)
    torch.testing.assert_close(first, longer[:, :2])
    torch.testing.assert_close(model.decode(first, 7), model.decode(longer, 7))


def test_late_loss_reaches_early_latent_and_all_modules(model):
    prompt, lengths = inputs()
    generated = []
    hook = model.latent_head.register_forward_hook(
        lambda module, args, output: generated.append(output)
    )
    logits = model(prompt, lengths, 16)
    hook.remove()
    generated[0].retain_grad()
    labels = torch.randint(3, model.cfg.vocab_size, (2, 4))
    F.cross_entropy(
        logits[:, -4:].reshape(-1, model.cfg.vocab_size), labels.flatten()
    ).backward()
    assert generated[0].grad.abs().sum() > 0
    for module in [model.encoder, model.backbone, model.latent_head, model.decoder]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_collate_has_no_teacher_forcing_input():
    batch = collate(
        [
            (np.array([4, 5, 6]), np.array([7, 8, 9])),
            (np.array([10]), np.array([2])),
        ],
        pad_id=0,
    )
    assert set(batch) == {"prompt", "prompt_lengths", "labels"}
    assert batch["labels"].tolist() == [[7, 8, 9], [2, -100, -100]]


def test_zero_latent_diagnostic_bypasses_all_prompt_computation(model):
    prompt, lengths = inputs()
    first = model(prompt, lengths, 8, use_latents=False)
    other = model(torch.full_like(prompt, 20), lengths, 8, use_latents=False)
    torch.testing.assert_close(first, other)
    first.square().mean().backward()
    for module in [model.encoder, model.backbone, model.latent_head]:
        assert all(p.grad is None for p in module.parameters())
    assert model.decoder.output.weight.grad is not None


def test_greedy_generation_matches_parallel_logits(model):
    prompt, lengths = inputs()
    logits = model(prompt, lengths, 7).detach().clone()
    logits[..., :3] = -torch.inf
    with torch.no_grad():
        # 排除随机初始化碰巧首 token 为 EOS，检查完整非整块长度。
        model.decoder.output.weight[model.cfg.eos_id].zero_()
    logits = model(prompt, lengths, 7).detach()
    logits[..., :3] = -torch.inf
    expected = logits.argmax(-1)
    actual = model.generate(prompt, lengths, max_new_tokens=7, min_new_tokens=7)
    assert torch.equal(actual, expected)


def test_eos_can_stop_inside_a_parallel_block(model, monkeypatch):
    def fixed_logits(latents):
        logits = torch.zeros(
            latents.shape[0], latents.shape[1], 4, model.cfg.vocab_size
        )
        logits[..., 2] = 100
        logits[..., 4] = 50
        return logits

    monkeypatch.setattr(model.decoder, "forward", fixed_logits)
    prompt, lengths = inputs()
    original = model.generate(prompt, lengths, max_new_tokens=7)
    assert original.tolist() == [[2], [2]]
    limited = model.generate(prompt, lengths, max_new_tokens=7, min_new_tokens=2)
    assert limited.tolist() == [[4, 4, 2], [4, 4, 2]]
    with pytest.raises(ValueError, match="min_new_tokens"):
        model.generate(prompt, lengths, max_new_tokens=4, min_new_tokens=5)


def test_fixed_batch_can_be_overfit_without_text_history(model):
    prompt, lengths = inputs()
    labels = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 2], [10, 9, 8, 7, 6, 5, 4, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        logits = model(prompt, lengths, labels.shape[1])
        loss = F.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size), labels.flatten()
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.4
