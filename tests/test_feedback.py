"""逐块反馈的结束、掩码与随机性检查。"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latent_ar.feedback import generate_with_feedback


class StubCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded = []

    def encode(self, ids, lengths):
        self.encoded.append((ids.clone(), lengths.clone()))
        mask = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
        value = (ids * mask).sum(dim=1).float()
        return value[:, None, None], torch.ones(len(ids), 1, dtype=torch.bool)

    def decode(self, latents):
        # z 的前两位是批次行号和步数。pad/bos 的分数故意最高。
        batch = latents.shape[0]
        logits = torch.zeros(batch, 16, 7)
        logits[:, :, 0] = 100
        logits[:, :, 1] = 90
        logits[:, :, 3] = 10
        logits[:, :, 2] = 9
        for row in range(batch):
            if int(latents[row, 0, 0]) == 0:
                logits[row, 1, 2] = 20
            elif int(latents[row, 0, 1]) >= 1:
                logits[row, 2, 2] = 20
        return logits


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, latent, positions, valid, cache=None):
        self.calls.append((latent.clone(), positions.clone(), valid.clone()))
        return torch.zeros(latent.shape[0], latent.shape[1], 2), []


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(chunk_size=16, pad_id=0, bos_id=1, eos_id=2)
        self.codec = StubCodec()
        self.backbone = StubBackbone()
        self.step = 0

    def generation_generators(self, generator, device):
        return generator, None

    def sample_latent(self, context, generator=None):
        result = torch.stack(
            (torch.arange(len(context)).float(), torch.full((len(context),), self.step)),
            dim=-1,
        )
        self.step += 1
        return result


def generate(model, **kwargs):
    return generate_with_feedback(
        model,
        torch.tensor([[4, 5], [4, 5]]),
        torch.tensor([2, 2]),
        return_trace=True,
        **kwargs,
    )


def test_mixed_eos_masks_finished_cache_and_encodes_only_emitted_tokens():
    model = StubModel()
    result = generate(model, max_new_tokens=32)
    assert model.training
    assert result["output_ids"].shape == (2, 19)
    assert result["output_ids"][0].tolist() == [3, 2] + [2] * 17
    assert result["output_ids"][1].tolist() == [3] * 18 + [2]
    first, second = result["records"]
    assert first["lengths"].tolist() == [2, 16]
    assert second["active"].tolist() == [False, True]
    assert second["lengths"].tolist() == [0, 3]
    assert model.codec.encoded[1][0][0].tolist() == [3, 2] + [0] * 14
    assert model.codec.encoded[2][1].tolist() == [3]
    assert model.backbone.calls[1][2].tolist() == [[True, False], [True, True]]
    torch.testing.assert_close(model.backbone.calls[1][0][:, 0], first["feedback"])
    assert all(not value.requires_grad for record in result["records"] for value in record.values())


def test_min_length_tail_and_raw_feedback_share_first_block():
    text_model, raw_model = StubModel(), StubModel()
    text = generate(text_model, max_new_tokens=19, min_new_tokens=19)
    raw = generate(raw_model, max_new_tokens=19, min_new_tokens=19, feedback_mode="raw")
    assert text["output_ids"].tolist() == [[3] * 19, [3] * 19]
    assert text["records"][-1]["lengths"].tolist() == [3, 3]
    torch.testing.assert_close(text["output_ids"][:, :16], raw["output_ids"][:, :16])
    torch.testing.assert_close(raw["records"][0]["feedback"], raw["records"][0]["z"])


def test_invalid_temperature_and_mode_restore():
    model = StubModel().eval()
    with pytest.raises(ValueError, match="temperature=0"):
        generate(model, temperature=0.5)
    assert not model.training
    generate(model, max_new_tokens=2)
    assert not model.training


def test_bfloat16_latent_accepts_float32_history_encoding():
    """生成头与历史编码不同类型时，文本反馈应保留编码精度与宽度。"""
    class HalfLatentModel(StubModel):
        def sample_latent(self, context, generator=None):
            return super().sample_latent(context, generator).to(torch.bfloat16)

    model = HalfLatentModel()
    result = generate(model, max_new_tokens=32)
    assert result["records"][0]["z"].dtype == torch.bfloat16
    assert result["records"][0]["feedback"].dtype == torch.float32
    assert model.backbone.calls[1][0].dtype == torch.float32
    assert result["records"][1]["active"].tolist() == [False, True]
