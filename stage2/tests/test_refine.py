import torch

from stage2.models import TwoStreamLM
from stage2.schedules import refine_queries, refine_slot, visibility

P, B, L = 4, 3, 8


def test_jump_is_first_written_slot():
    for g in (1, 2, 4):
        v = visibility("reverse", P, B, L, g, g)
        jumps, _, _ = refine_queries(P, B, L, refine_slot("reverse", L, g))
        for b in range(B):
            block = range(P + b * L, P + (b + 1) * L)
            # 最先写的词不读本块任何词。
            first = [i for i in block if not v[i, P + b * L : P + (b + 1) * L].any()]
            assert first == [int(jumps[b])]


def test_refine_rows_see_own_block_but_no_jump_or_right_block():
    jumps, rows, content = refine_queries(P, B, L, L - 4)
    for b in range(B):
        expected = torch.zeros(P + B * L, dtype=torch.bool)
        expected[: P + (b + 1) * L] = True
        expected[jumps] = False
        assert torch.equal(rows[b], expected)
    assert not content[:, jumps].any()


def test_refine_query_depends_exactly_on_its_rows():
    torch.manual_seed(0)
    n = P + B * L
    model = TwoStreamLM(50, n, 32, 2, 2, 64).double().eval()
    jumps, rows, content = refine_queries(P, B, L, L - 4)
    ids = torch.randint(0, 50, (1, n))
    words = ids.repeat(n + 1, 1)
    for position in range(n):
        words[position + 1, position] = (words[position + 1, position] + 1) % 50
    with torch.no_grad():
        logits = model(words, content, jumps, rows)
    change = (logits[1:] - logits[:1]).abs().amax(-1)  # [改动的输入, 查询]
    # 跳跃词本身（以及其他块的跳跃词）通过任何路径都影响不到重解码查询。
    # 不可见输入只留下 1e-17 量级的浮点舍入；真实依赖在 1e-3 以上。
    assert change[~rows.T].max() < 1e-12
    assert (change[rows.T] > 1e-6).all()


def test_default_query_rows_unchanged():
    torch.manual_seed(0)
    n = P + B * L
    model = TwoStreamLM(50, n, 32, 2, 2, 64).double().eval()
    v = visibility("reverse", P, B, L, 4, 4)
    targets = torch.arange(P, n)
    ids = torch.randint(0, 50, (2, n))
    with torch.no_grad():
        assert torch.equal(model(ids, v, targets), model(ids, v, targets, v[targets]))


def test_forward_refines_block_head():
    assert refine_slot("forward", L) == 0
    jumps, rows, _ = refine_queries(P, B, L, 0)
    assert jumps.tolist() == [P, P + L, P + 2 * L]
    # 块首重解码读到上一块全部词和本块其余词。
    assert rows[1, P : P + 2 * L].sum() == 2 * L - 2
