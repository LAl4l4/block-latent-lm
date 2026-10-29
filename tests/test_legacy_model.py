import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from latent_ar.data import collate, crop_story
from latent_ar.legacy_model import LatentARModel, ModelConfig


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = ModelConfig(
        vocab_size=32,
        encoder_dim=8,
        dim=16,
        layers=2,
        heads=2,
        ff_dim=32,
        decoder_dim=16,
        decoder_layers=2,
        decoder_heads=2,
        decoder_ff_dim=32,
    )
    return LatentARModel(cfg)


def example_inputs():
    prompt = torch.tensor([[4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 0, 0, 0, 0]])
    lengths = torch.tensor([7, 3])
    decoder_input = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10], [1, 10, 9, 8, 7, 6, 5, 4]])
    return prompt, lengths, decoder_input


def test_padding_does_not_change_prompt_or_rollout(model):
    prompt, lengths, _ = example_inputs()
    memory, _, valid, _ = model.rollout(prompt, lengths, 3)
    padded = F.pad(prompt, (0, 6), value=31)
    other, _, other_valid, _ = model.rollout(padded, lengths, 3)
    torch.testing.assert_close(memory[:, -3:], other[:, -3:], atol=2e-6, rtol=2e-5)
    assert valid.sum(-1).tolist() == other_valid.sum(-1).tolist() == [2, 1]
    for i, length in enumerate(lengths.tolist()):
        alone, _, _, _ = model.rollout(
            prompt[i : i + 1, :length], lengths[i : i + 1], 3
        )
        torch.testing.assert_close(memory[i, -3:], alone[0, -3:], atol=2e-6, rtol=2e-5)


def test_decoder_cannot_see_future_teacher_tokens(model):
    prompt, lengths, decoder_input = example_inputs()
    original = model(prompt, lengths, decoder_input)
    changed = decoder_input.clone()
    changed[:, 4:] = 19
    perturbed = model(prompt, lengths, changed)
    torch.testing.assert_close(original[:, :4], perturbed[:, :4], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(original[:, 4:], perturbed[:, 4:])


def test_decoder_cannot_see_future_latents(model):
    prompt, lengths, decoder_input = example_inputs()
    memory, positions, valid, _ = model.rollout(prompt, lengths, 2)
    first, _ = model.decoder(
        decoder_input, valid, 2, model.decoder.project_memory(memory, positions)
    )
    changed = memory.clone()
    changed[:, -1] = torch.randn_like(changed[:, -1]) * 10
    second, _ = model.decoder(
        decoder_input, valid, 2, model.decoder.project_memory(changed, positions)
    )
    torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(first[:, 4:], second[:, 4:])


def test_cached_backbone_matches_full_prefix(model):
    x = torch.randn(2, 6, model.cfg.dim)
    positions = torch.arange(6).expand(2, -1)
    valid = torch.ones(2, 6, dtype=torch.bool)
    full, _ = model.backbone(x, positions, valid)
    prefix, cache = model.backbone(x[:, :3], positions[:, :3], valid[:, :3])
    pieces = [prefix]
    for i in range(3, 6):
        part, cache = model.backbone(
            x[:, i : i + 1], positions[:, i : i + 1], valid[:, : i + 1], cache
        )
        pieces.append(part)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-5)


def test_cached_decoder_matches_teacher_forcing(model):
    prompt, lengths, decoder_input = example_inputs()
    memory, positions, valid, _ = model.rollout(prompt, lengths, 2)
    memory_kvs = model.decoder.project_memory(memory, positions)
    full, _ = model.decoder(decoder_input, valid, 2, memory_kvs)
    cache, pieces = None, []
    for i in range(decoder_input.shape[1]):
        part, cache = model.decoder(
            decoder_input[:, i : i + 1], valid, 2, memory_kvs, cache
        )
        pieces.append(part)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-5)


def test_last_latent_backpropagates_through_earlier_steps(model):
    prompt, lengths, _ = example_inputs()
    _, _, _, generated = model.rollout(prompt, lengths, 4)
    generated[0].retain_grad()
    generated[-1][:, 0].sum().backward()
    assert generated[0].grad is not None
    assert generated[0].grad.abs().sum() > 0
    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert model.backbone.blocks[0].attention.qkv.weight.grad.abs().sum() > 0


def test_joint_ce_updates_all_three_modules(model):
    prompt, lengths, decoder_input = example_inputs()
    labels = torch.randint(3, model.cfg.vocab_size, decoder_input.shape)
    logits = model(prompt, lengths, decoder_input)
    F.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_size), labels.flatten()
    ).backward()
    for module in [model.encoder, model.backbone, model.latent_head, model.decoder]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_no_latent_ablation_bypasses_backbone(model):
    prompt, lengths, decoder_input = example_inputs()
    logits = model(prompt, lengths, decoder_input, use_latents=False)
    logits.square().mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.encoder.embedding.weight.grad is not None


def test_natural_eos_and_truncation():
    story = np.asarray(list(range(3, 30)) + [2], dtype=np.uint32)
    observed_eos, observed_truncation = False, False
    for seed in range(100):
        prompt, target = crop_story(story, random.Random(seed), 5, 7, 4)
        boundary = int(prompt[-1]) - 3 + 1
        assert target.tolist() == story[boundary : boundary + 4].tolist()
        assert 2 not in prompt
        observed_eos |= target[-1] == 2
        observed_truncation |= target[-1] != 2
    assert observed_eos and observed_truncation


def test_collation_keeps_prompt_boundary_and_ignores_padding():
    batch = collate(
        [(np.array([4, 5, 6]), np.array([7, 8, 9])), (np.array([10]), np.array([2]))],
        pad_id=0,
    )
    assert batch["prompt"].tolist() == [[4, 5, 6], [10, 0, 0]]
    assert "decoder_input" not in batch
    assert batch["labels"].tolist() == [[7, 8, 9], [2, -100, -100]]


def test_tiny_fixed_batch_can_overfit(model):
    prompt, lengths, decoder_input = example_inputs()
    labels = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 2], [10, 9, 8, 7, 6, 5, 4, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(35):
        optimizer.zero_grad()
        logits = model(prompt, lengths, decoder_input)
        loss = F.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size), labels.flatten()
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.4


def test_minimum_generation_length_is_explicit_and_optional(model, monkeypatch):
    """让 decoder 强烈偏向 EOS，确认默认停止与显式长度限制均正确。"""

    def prefer_eos(ids, *arguments, **kwargs):
        logits = torch.zeros(ids.shape[0], ids.shape[1], model.cfg.vocab_size)
        logits[..., model.cfg.eos_id] = 100
        logits[..., 4] = 50
        return logits, None

    monkeypatch.setattr(model.decoder, "forward", prefer_eos)
    prompt = torch.tensor([[4, 5, 6]])
    lengths = torch.tensor([3])
    original = model.generate(prompt, lengths, max_new_tokens=6, temperature=0)
    assert original.tolist() == [[model.cfg.eos_id]]

    constrained = model.generate(
        prompt, lengths, max_new_tokens=6, temperature=0, min_new_tokens=3
    )
    assert constrained.tolist() == [[4, 4, 4, model.cfg.eos_id]]


def test_generation_rejects_impossible_minimum_length(model):
    with pytest.raises(ValueError, match="min_new_tokens"):
        model.generate(
            torch.tensor([[4, 5]]),
            torch.tensor([2]),
            max_new_tokens=4,
            min_new_tokens=5,
        )
