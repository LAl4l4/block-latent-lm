"""学生反馈目标、状态恢复、EOS 和共享采样噪声检查。"""

import pytest
import torch

from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel
from latent_ar.feedback_student import (
    FeedbackStudent,
    StudentConfig,
    feedback_statistics,
    generate_student_feedback,
)


def teacher():
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
        token_temperature=0.0,
        noise_dim=4,
        energy_layers=1,
        energy_candidates=2,
        target_samples=2,
        energy_batch_blocks=2,
    )
    return CalmTokenModel(cfg)


def student(noise_dim=0):
    return FeedbackStudent(
        StudentConfig(
            context_dim=16,
            noise_dim=noise_dim,
            hidden_dim=16,
            layers=2,
            feedback_dim=16,
        )
    )


def test_loss_active_mask_and_checkpoint():
    current = student(4)
    context, noise = torch.randn(3, 16), torch.rand(3, 4)
    prediction = current(context, noise)
    target = torch.randn(3, 16)
    stats = feedback_statistics(prediction, target, torch.tensor([True, False, True]))
    expected = (prediction[[0, 2]] - target[[0, 2]]).square().mean()
    torch.testing.assert_close(stats["loss"], expected)
    stats["loss"].backward()
    assert current.context.weight.grad.abs().sum() > 0
    assert not stats["cosine"].requires_grad
    restored = FeedbackStudent.from_checkpoint(current.checkpoint())
    torch.testing.assert_close(prediction, restored(context, noise))
    with pytest.raises(ValueError):
        current(context, None)


def test_mixed_eos_tail_actual_feedback_cache_and_restore():
    base, learner = teacher().train(), student().train()
    original_decode = base.codec.decode
    cache_masks = []
    handle = base.backbone.register_forward_pre_hook(
        lambda module, args: cache_masks.append(args[2].clone())
    )

    def force_decode(z):
        logits = torch.zeros(z.shape[0], 4, 32)
        logits[:, :, 0] = 100
        logits[:, :, 1] = 90
        logits[:, :, 3] = 10
        logits[0, 1, 2] = 20
        return logits

    base.codec.decode = force_decode
    prompt = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    lengths = torch.tensor([4, 4])
    try:
        result = generate_student_feedback(
            base,
            learner,
            prompt,
            lengths,
            max_new_tokens=6,
            generator=torch.Generator().manual_seed(12),
            return_trace=True,
        )
    finally:
        handle.remove()
        base.codec.decode = original_decode
    assert base.training and learner.training and not base.codec.training
    first, second = result["records"]
    assert first["lengths"].tolist() == [2, 4]
    assert second["active"].tolist() == [False, True]
    assert second["lengths"].tolist() == [0, 2]
    assert result["output_ids"].tolist() == [[3, 2, 2, 2, 2, 2], [3] * 6]
    assert cache_masks[1].tolist() == [[True, False], [True, True]]
    original_tokens = torch.tensor([[3, 2, 0, 0], [3, 3, 3, 3]])
    expected, _ = base.encode_history(original_tokens, torch.tensor([2, 4]))
    torch.testing.assert_close(first["true_feedback"], expected[:, 0])
    assert all(
        not value.requires_grad
        for record in result["records"]
        for value in record.values()
    )


def test_temperature_policy_and_same_u():
    base, learner = teacher(), student()
    base.cfg.token_temperature = 0.8
    prompt, lengths = torch.tensor([[1, 4, 5, 6]]), torch.tensor([4])
    with pytest.raises(ValueError, match="token_temperature=0"):
        generate_student_feedback(base, learner, prompt, lengths)
    stochastic = student(4)
    result = generate_student_feedback(
        base,
        stochastic,
        prompt,
        lengths,
        max_new_tokens=4,
        generator=torch.Generator().manual_seed(55),
        return_trace=True,
    )
    record = result["records"][0]
    with torch.no_grad():
        expected = stochastic(record["context"], record["noise"])
        target_z = base.sample_latent(record["context"], noise=record["noise"])
    torch.testing.assert_close(record["predicted_feedback"], expected)
    torch.testing.assert_close(record["z"], target_z)


def test_attention_window_boundaries_and_checkpoint():
    from latent_ar.feedback_student import (
        AttentionFeedbackStudent,
        AttentionStudentConfig,
        build_history_windows,
        load_feedback_student,
    )

    data = {
        "h": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "story_id": torch.tensor([1, 2, 1, 2, 1, 1]),
        "block_index": torch.tensor([0, 0, 1, 1, 2, 3]),
    }
    indices, valid = build_history_windows(data, 2)
    broken = dict(data, block_index=torch.tensor([0, 0, 1, 1, 4, 3]))
    with pytest.raises(ValueError, match="连续"):
        build_history_windows(broken, 2)
    assert indices[4].tolist() == [0, 2, 4]
    assert indices[5].tolist() == [2, 4, 5]
    assert valid[0].tolist() == [False, False, True]
    assert indices[3, -2:].tolist() == [1, 3]
    cfg = AttentionStudentConfig(
        context_dim=4, feedback_dim=4, hidden_dim=8, heads=2, history_blocks=2
    )
    model = AttentionFeedbackStudent(cfg).eval()
    window = data["h"][indices[[0, 4]]]
    mask = valid[[0, 4]]
    with torch.no_grad():
        out = model(window, valid=mask)
        changed = window.clone()
        changed[0, :2] = 1e20
        torch.testing.assert_close(out, model(changed, valid=mask))
        data["h"][5] = -999  # A future state cannot affect block 2.
        torch.testing.assert_close(
            out[1], model(data["h"][indices[[4]]], valid=mask[[1]])[0]
        )
        restored = load_feedback_student(model.checkpoint()).eval()
        torch.testing.assert_close(out, restored(window, valid=mask))


def test_attention_closed_loop_first_block_and_mixed_eos():
    from latent_ar.feedback_student import (
        AttentionFeedbackStudent,
        AttentionStudentConfig,
    )

    base = teacher().eval()
    learner = AttentionFeedbackStudent(
        AttentionStudentConfig(
            context_dim=16, feedback_dim=16, hidden_dim=16, heads=2, history_blocks=3
        )
    ).eval()
    prompt = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    lengths = torch.tensor([4, 4])

    def force_decode(z):
        logits = torch.zeros(z.shape[0], 4, 32, device=z.device)
        logits[:, :, 3] = 10
        logits[0, 1, 2] = 20
        return logits

    base.codec.decode = force_decode
    seed = 12
    result = generate_student_feedback(
        base,
        learner,
        prompt,
        lengths,
        max_new_tokens=6,
        generator=torch.Generator().manual_seed(seed),
        return_trace=True,
    )
    first, second = result["records"]
    teacher_output = base.generate(
        prompt, lengths, max_new_tokens=6, generator=torch.Generator().manual_seed(seed)
    )
    torch.testing.assert_close(result["output_ids"][:, :4], teacher_output[:, :4])
    assert second["active"].tolist() == [False, True]
    with torch.no_grad():
        window = torch.zeros(2, 4, 16)
        window[:, -1] = first["context"]
        valid = torch.tensor([[False, False, False, True]]).expand(2, -1)
        torch.testing.assert_close(
            first["predicted_feedback"], learner(window, valid=valid)
        )


def test_prompt_history_matches_prefill_and_generation_for_mixed_lengths():
    from contextlib import nullcontext
    from experiments.distill_gru_feedback import extract_prompt_history
    from latent_ar.feedback_student import (
        AttentionFeedbackStudent,
        AttentionStudentConfig,
        materialize_history_windows,
    )

    base = teacher().eval()
    prompts = [
        torch.tensor([1, 4, 5, 6]),
        torch.tensor([1, 7, 8, 9, 10, 11, 12, 13, 14]),
    ]
    prefix, prefix_valid = extract_prompt_history(
        base, prompts, 2, 2, torch.device("cpu"), nullcontext
    )
    assert prefix_valid.tolist() == [[False, False], [True, True]]
    padded = torch.nn.utils.rnn.pad_sequence(
        prompts, batch_first=True, padding_value=base.cfg.pad_id
    )
    lengths = torch.tensor([len(p) for p in prompts])
    encoded, valid = base.encode_history(padded, lengths)
    hidden, _ = base.backbone(
        encoded, torch.arange(encoded.shape[1]).expand(2, -1), valid
    )
    current = hidden[torch.arange(2), valid.sum(-1) - 1].detach()
    data = {
        "h": current,
        "story_id": torch.tensor([42, 7]),
        "block_index": torch.tensor([0, 0]),
        "prompt_story_ids": torch.tensor([7, 42]),
        "prefix_h": prefix.flip(0),
        "prefix_valid": prefix_valid.flip(0),
    }
    window, mask = materialize_history_windows(data, 2, True)
    torch.testing.assert_close(window[:, :-1], prefix)
    assert torch.equal(mask[:, :-1], prefix_valid)
    learner = AttentionFeedbackStudent(
        AttentionStudentConfig(
            context_dim=16,
            feedback_dim=16,
            hidden_dim=16,
            heads=2,
            history_blocks=2,
            use_prompt_history=True,
        )
    ).eval()
    captured = []
    handle = learner.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(
            (args[0].detach().clone(), kwargs["valid"].detach().clone())
        ),
        with_kwargs=True,
    )
    try:
        generate_student_feedback(
            base,
            learner,
            padded,
            lengths,
            max_new_tokens=4,
            generator=torch.Generator().manual_seed(9),
        )
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0][0], window)
    assert torch.equal(captured[0][1], mask)
    with pytest.raises(ValueError, match="缺失"):
        materialize_history_windows(
            {k: v for k, v in data.items() if k != "prefix_h"}, 2, True
        )
    old = learner.checkpoint()
    old["config"].pop("use_prompt_history")
    assert not AttentionFeedbackStudent.from_checkpoint(old).cfg.use_prompt_history
