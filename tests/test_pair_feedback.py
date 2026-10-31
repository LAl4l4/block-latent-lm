import pytest
import torch

from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel
from latent_ar.feedback import generate_with_feedback
from latent_ar.feedback_student import (
    FeedbackStudent,
    StudentConfig,
    AttentionFeedbackStudent,
    AttentionStudentConfig,
)
from latent_ar.pair_feedback import generate_pair_feedback


def models():
    torch.manual_seed(8)
    cfg = CalmTokenConfig(
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
        token_head_dim=8,
        token_temperature=0,
        noise_dim=4,
        energy_layers=1,
        energy_candidates=2,
        target_samples=2,
        energy_batch_blocks=2,
    )
    return CalmTokenModel(cfg), FeedbackStudent(
        StudentConfig(context_dim=16, hidden_dim=16, feedback_dim=16)
    )


def test_replay_batch_and_partial_budget():
    teacher, student = models()
    teacher.train()
    student.train()
    prompt = torch.tensor([[1, 4, 5, 6, 7], [1, 8, 9, 0, 0]])
    lengths = torch.tensor([5, 3])
    kwargs = dict(
        max_new_tokens=13, min_new_tokens=13, return_trace=True, verify_repair=True
    )
    serial = generate_pair_feedback(
        teacher,
        student,
        prompt,
        lengths,
        generator=torch.Generator().manual_seed(5),
        **kwargs
    )
    batched = generate_pair_feedback(
        teacher,
        student,
        prompt,
        lengths,
        generator=torch.Generator().manual_seed(5),
        parallel="batch",
        **kwargs
    )
    assert teacher.training and student.training
    assert serial["output_ids"].shape == (2, 13)
    assert torch.equal(serial["output_ids"], batched["output_ids"])
    assert serial["records"][1]["max_cache_error"] < 1e-5
    assert serial["records"][1]["repair_valid"].tolist() == [[True, True], [True, True]]
    assert serial["records"][1]["repair_positions"].tolist() == [[2, 3], [1, 2]]
    baseline = generate_with_feedback(
        teacher,
        prompt,
        lengths,
        max_new_tokens=4,
        min_new_tokens=4,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.equal(serial["output_ids"][:, :4], baseline)


def test_eos_and_attention_rejected():
    teacher, student = models()

    def decode(z):
        result = torch.zeros(z.shape[0], 4, 32)
        result[:, :, 3] = 10
        result[0, 1, 2] = 20
        return result

    teacher.codec.decode = decode
    prompt = torch.tensor([[1, 4], [1, 5]])
    result = generate_pair_feedback(
        teacher,
        student,
        prompt,
        torch.tensor([2, 2]),
        max_new_tokens=10,
        return_trace=True,
    )
    assert result["records"][0]["lengths"].tolist() == [2, 4]
    assert result["records"][1]["active"].tolist() == [False, True]
    assert result["records"][1]["repair_valid"].tolist() == [
        [False, False],
        [True, True],
    ]
    assert result["output_ids"].tolist() == [[3, 2] + [2] * 8, [3] * 10]
    attention = AttentionFeedbackStudent(
        AttentionStudentConfig(context_dim=16, feedback_dim=16, hidden_dim=16, heads=2)
    )
    with pytest.raises(ValueError, match="MLP"):
        generate_pair_feedback(teacher, attention, prompt, torch.tensor([2, 2]))


def test_oracle_feedback_matches_exact_and_weights_unchanged():
    teacher, student = models()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}

    def oracle(context, noise=None):
        z = teacher.sample_latent(context, noise=torch.zeros(context.shape[0], 4))
        scores = teacher.codec.decode(z[:, None]).float()
        scores[..., :3] = -torch.inf
        ids = scores.argmax(-1)
        encoded, _ = teacher.encode_history(ids, torch.full((len(ids),), 4))
        return encoded[:, 0]

    student.forward = oracle
    prompt = torch.tensor([[1, 4, 5, 6, 7], [1, 8, 9, 0, 0]])
    lengths = torch.tensor([5, 3])
    actual = generate_pair_feedback(
        teacher,
        student,
        prompt,
        lengths,
        max_new_tokens=12,
        min_new_tokens=12,
        generator=torch.Generator().manual_seed(9),
    )
    expected = generate_with_feedback(
        teacher,
        prompt,
        lengths,
        max_new_tokens=12,
        min_new_tokens=12,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.equal(actual, expected)
    assert all(
        torch.equal(value, teacher.state_dict()[key]) for key, value in before.items()
    )
