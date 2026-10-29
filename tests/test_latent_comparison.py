"""确认均值对照只改生成 latent，且复用原采样器不会引入额外差异。"""

import pytest
import torch

from compare_latent_generation import mean_generated_latents, use_fixed_rollout
from latent_ar.model import LatentARModel, ModelConfig


def test_mean_comparison_preserves_prompt_and_input_memory():
    memory = torch.arange(2 * 7 * 4, dtype=torch.float32).view(2, 7, 4)
    original = memory.clone()
    changed = mean_generated_latents(memory, 3)
    torch.testing.assert_close(changed[:, :4], original[:, :4])
    torch.testing.assert_close(memory, original)
    expected = original[:, -3:].mean(dim=1, keepdim=True).expand(-1, 3, -1)
    torch.testing.assert_close(changed[:, -3:], expected)


def test_fixed_memory_matches_regular_generation_and_restores_rollout():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = LatentARModel(
        ModelConfig(
            vocab_size=32,
            encoder_dim=8,
            dim=16,
            layers=1,
            heads=2,
            ff_dim=32,
            decoder_dim=16,
            decoder_layers=1,
            decoder_heads=2,
            decoder_ff_dim=32,
        )
    ).eval()
    prompt = torch.tensor([[4, 5, 6]])
    lengths = torch.tensor([3])
    expected = model.generate(
        prompt,
        lengths,
        max_new_tokens=8,
        temperature=0.8,
        generator=torch.Generator().manual_seed(42),
    )
    with torch.inference_mode():
        rollout = model.rollout(prompt, lengths, 2)
    original_method = model.rollout
    with use_fixed_rollout(model, rollout):
        actual = model.generate(
            prompt,
            lengths,
            max_new_tokens=8,
            temperature=0.8,
            generator=torch.Generator().manual_seed(42),
        )
    assert torch.equal(actual, expected)
    assert model.rollout == original_method

    with pytest.raises(RuntimeError, match="test"):
        with use_fixed_rollout(model, rollout):
            raise RuntimeError("test")
    assert model.rollout == original_method
