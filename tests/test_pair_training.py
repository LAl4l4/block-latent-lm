"""两块前瞻训练前向：与原教师一致、因果隔离、与推理时缓存调度一致。"""

import torch

from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel
from latent_ar.feedback_student import FeedbackStudent, StudentConfig
from latent_ar.pair_training import (
    history_layout,
    make_constant,
    pair_contexts,
    pair_statistics,
)


def models():
    torch.manual_seed(11)
    cfg = CalmTokenConfig(
        vocab_size=32, chunk_size=4, encoder_dim=8, dim=16, layers=2, heads=2,
        ff_dim=32, decoder_dim=8, decoder_layers=1, decoder_heads=2,
        decoder_ff_dim=16, token_head_dim=8, token_temperature=0, noise_dim=4,
        energy_layers=1, energy_candidates=2, target_samples=2, energy_batch_blocks=2,
    )
    student = FeedbackStudent(StudentConfig(context_dim=16, hidden_dim=16, feedback_dim=16))
    return CalmTokenModel(cfg).eval(), student.eval()


def inputs():
    # 两行 prompt 长度不同（含不完整尾块），目标分别 4 块和 2 块。
    prompt = torch.tensor([[1, 4, 5, 6, 7, 8], [1, 9, 10, 0, 0, 0]])
    lengths = torch.tensor([6, 3])
    labels = torch.tensor([
        list(range(3, 19)),
        [20, 21, 22, 23, 24, 25, 2] + [-100] * 9,
    ])
    return prompt, lengths, labels


def test_sync_matches_original_teacher_objective():
    model, student = models()
    prompt, lengths, labels = inputs()
    with torch.no_grad():
        ours = pair_statistics(model, student, prompt, lengths, labels, "sync")
        original = model.prior_statistics(prompt, lengths, labels)
        layout = history_layout(model, prompt, lengths, labels)
        contexts, _ = pair_contexts(model, student, layout, "sync")
        reference, *_ = model.teacher_contexts(prompt, lengths, labels)
    torch.testing.assert_close(ours["objective_sum"], original["objective_sum"])
    assert ours["tokens"] == original["tokens"]
    torch.testing.assert_close(contexts, reference)
    assert ours["even_tokens"] + ours["odd_tokens"] == ours["tokens"]


def test_even_blocks_unchanged_and_odd_blocks_use_student():
    model, student = models()
    prompt, lengths, labels = inputs()
    with torch.no_grad():
        layout = history_layout(model, prompt, lengths, labels)
        sync, _ = pair_contexts(model, student, layout, "sync")
        pair, predictions = pair_contexts(model, student, layout, "state")
    torch.testing.assert_close(pair[:, 0::2], sync[:, 0::2])
    assert not torch.allclose(pair[0, 1], sync[0, 1])
    assert not torch.allclose(pair[0, 3], sync[0, 3])
    assert sorted(predictions) == [0, 2]


def test_odd_block_cannot_see_previous_block_text():
    model, student = models()
    prompt, lengths, labels = inputs()
    changed = labels.clone()
    changed[0, 8:12] = torch.tensor([27, 28, 29, 30])  # 第 2 块（偶数）文字
    changed[0, 0:4] = torch.tensor([26, 25, 24, 23])  # 第 0 块（偶数）文字
    with torch.no_grad():
        a, _ = pair_contexts(model, student, history_layout(model, prompt, lengths, labels), "state")
        b, _ = pair_contexts(model, student, history_layout(model, prompt, lengths, changed), "state")
        sa, _ = pair_contexts(model, student, history_layout(model, prompt, lengths, labels), "sync")
        sb, _ = pair_contexts(model, student, history_layout(model, prompt, lengths, changed), "sync")
    # 块1只依赖 prompt；块3不依赖块2文字（但依赖块0、块1真实文字）。
    torch.testing.assert_close(a[0, 1], b[0, 1])
    assert not torch.allclose(a[0, 2], b[0, 2])  # 块2读取块0、块1真实反馈
    assert not torch.allclose(a[0, 3], b[0, 3])  # 块3仍读取块0真实反馈
    assert not torch.allclose(sa[0, 1], sb[0, 1])  # 同步教师块1读取块0文字

    only_block2 = labels.clone()
    only_block2[0, 8:12] = torch.tensor([27, 28, 29, 30])
    with torch.no_grad():
        c, _ = pair_contexts(model, student, history_layout(model, prompt, lengths, only_block2), "state")
        sc, _ = pair_contexts(model, student, history_layout(model, prompt, lengths, only_block2), "sync")
    torch.testing.assert_close(a[0, 3], c[0, 3])
    assert not torch.allclose(sa[0, 3], sc[0, 3])


def test_training_contexts_match_incremental_pair_schedule():
    """按 pair_feedback 的缓存调度逐组推进，结果应与一次性训练前向一致。"""
    model, student = models()
    prompt, lengths, labels = inputs()
    with torch.no_grad():
        layout = history_layout(model, prompt, lengths, labels)
        trained, _ = pair_contexts(model, student, layout, "state")
        feedback = layout["target_feedback"]
        for row in range(prompt.shape[0]):
            p, n = prompt[row : row + 1, : lengths[row]], lengths[row : row + 1]
            encoded, valid = model.encode_history(p, n)
            width = encoded.shape[1]
            positions = torch.arange(width)[None]
            hidden, cache = model.backbone(encoded, positions, valid)
            context = hidden[:, -1]
            n_blocks = int(layout["target_valid"][row].sum())
            for start in range(0, n_blocks, 2):
                torch.testing.assert_close(context[0], trained[row, start], rtol=1e-4, atol=1e-5)
                if start + 1 < n_blocks:
                    q = student(context)
                    provisional, _ = model.backbone(
                        q[:, None], torch.tensor([[width + start]]),
                        torch.cat((valid, torch.ones(1, 1, dtype=torch.bool)), 1), cache,
                    )
                    torch.testing.assert_close(
                        provisional[0, 0], trained[row, start + 1], rtol=1e-4, atol=1e-5
                    )
                count = min(2, n_blocks - start)
                if start + count >= n_blocks:
                    break
                inputs_ = feedback[row : row + 1, start : start + count]
                valid = torch.cat((valid, torch.ones(1, count, dtype=torch.bool)), 1)
                hidden, cache = model.backbone(
                    inputs_, (width + start + torch.arange(count))[None], valid, cache
                )
                context = hidden[:, -1]


def test_gradients_reach_student_and_backbone():
    model, student = models()
    model.train()
    student.train()
    prompt, lengths, labels = inputs()
    values = pair_statistics(model, student, prompt, lengths, labels, "state")
    values["objective_sum"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.codec.parameters())
    # predictions[0] 两行都有效（两行都有块1）；predictions[2] 只有行0有块3。
    assert values["q_count"] == 3


def test_constant_student_ignores_input():
    _, student = models()
    make_constant(student)
    with torch.no_grad():
        student.output.bias.normal_()
    a, b = student(torch.randn(3, 16)), student(torch.randn(3, 16))
    torch.testing.assert_close(a, b)
    trainable = [n for n, p in student.named_parameters() if p.requires_grad]
    assert trainable == ["output.bias"]


def test_shuffled_history_matches_teacher_shuffle_protocol():
    model, student = models()
    prompt, lengths, labels = inputs()
    with torch.no_grad():
        ours = pair_statistics(
            model, student, prompt, lengths, labels, "sync", shuffle_context=True
        )
        original = model.prior_statistics(prompt, lengths, labels, shuffle_context=True)
        plain = pair_statistics(model, student, prompt, lengths, labels, "sync")
    torch.testing.assert_close(ours["objective_sum"], original["objective_sum"])
    assert ours["odd_tokens"] == plain["odd_tokens"]
    assert ours["odd_nll"] != plain["odd_nll"]
