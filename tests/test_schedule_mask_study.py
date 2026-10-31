"""调度可见关系：因果基线、无泄漏、传递性，以及倒序的距离保证。"""

import torch

from experiments.schedule_mask_study import TwoStreamLM, visibility

PROMPT, BLOCKS, L = 3, 4, 8


def dropped_distances(visible):
    """两个位置都看不到对方的词对之间的距离。"""
    n = visible.shape[0]
    i, k = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    lost = (i > k) & ~visible & ~visible.T
    return (i - k)[lost]


def test_causal_is_lower_triangular():
    visible = visibility("causal", PROMPT, BLOCKS, L)
    n = PROMPT + BLOCKS * L
    assert torch.equal(visible, torch.ones(n, n, dtype=torch.bool).tril(-1))


def test_schedules_are_transitive_and_acyclic():
    for schedule, stagger in [
        ("causal", None), ("delayed", None), ("forward", 4), ("reverse", 4),
        ("reverse", 1), ("forward", 8), ("reverse", 8),
        ("reverse", 0), ("forward", 0), ("landing", None),
    ]:
        v = visibility(schedule, PROMPT, BLOCKS, L, stagger).int()
        assert not torch.diagonal(v).any()
        assert not (v & v.T).any()
        # 读到的位置所读到的一切，自己也能读到：查询流不会从内容流间接泄漏。
        assert ((v @ v) > 0).le(v.bool()).all(), (schedule, stagger)


def test_reverse_drops_only_distant_pairs_forward_drops_adjacent():
    for s in (1, 2, 4):
        assert dropped_distances(visibility("reverse", PROMPT, BLOCKS, L, s)).min() >= L + s
        assert dropped_distances(visibility("forward", PROMPT, BLOCKS, L, s)).min() == 1
    assert dropped_distances(visibility("delayed", PROMPT, BLOCKS, L)).min() == 1
    # 错开整块时倒序只改变块内顺序，不丢任何依赖。
    assert len(dropped_distances(visibility("reverse", PROMPT, BLOCKS, L, L))) == 0


def test_reverse_first_token_sees_previous_block_tail():
    v = visibility("reverse", PROMPT, BLOCKS, L, L // 2)
    first = PROMPT + 2 * L  # 第 2 个生成块的首词
    assert v[first, PROMPT + L : first].all()  # 上一块全部可见
    assert v[first, first + 1 : first + L].all()  # 首词最后生成，本块其余词都已可见


def test_prediction_ignores_invisible_tokens():
    torch.manual_seed(0)
    n = PROMPT + BLOCKS * L
    model = TwoStreamLM(32, n, 16, 2, 2, 32).eval()
    v = visibility("reverse", PROMPT, BLOCKS, L, 2)
    targets = torch.arange(PROMPT, n)
    ids = torch.randint(0, 32, (1, n))
    base = model(ids, v, targets)
    for position in range(n):
        changed = ids.clone()
        changed[0, position] = (changed[0, position] + 1) % 32
        out = model(changed, v, targets)
        unaffected = ~v[targets, position]
        torch.testing.assert_close(out[0, unaffected], base[0, unaffected])


def test_fully_parallel_reverse_keeps_distance_guarantee():
    v = visibility("reverse", PROMPT, BLOCKS, L, 0)
    assert dropped_distances(v).min() >= L
    assert dropped_distances(visibility("forward", PROMPT, BLOCKS, L, 0)).min() == 1
    # 第 k 块第 j 个词看到上一块中位置大于 j 的词（0 起编号）。
    j = 2
    row = PROMPT + 2 * L + j
    previous = torch.arange(PROMPT + L, PROMPT + 2 * L)
    assert torch.equal(v[row, previous], torch.arange(L) > j)


def test_landing_reads_only_next_block_first_token():
    v = visibility("landing", PROMPT, BLOCKS, L)
    base = visibility("forward", PROMPT, BLOCKS, L, 0)
    rows, keys = (v & ~base).nonzero(as_tuple=True)
    assert len(rows) > 0
    assert ((keys - PROMPT) % L == 0).all()
    assert ((keys - PROMPT) // L == (rows - PROMPT) // L + 1).all()


def test_sampling_rounds_match_schedule_parallelism():
    from experiments.schedule_sample_eval import rounds

    def steps(schedule, stagger=None):
        return int(rounds(visibility(schedule, 16, 7, 16, stagger), 16).max()) + 1

    assert steps("causal") == 112
    assert steps("reverse", 16) == 112
    assert steps("reverse", 8) == steps("forward", 8) == 6 * 8 + 16
    assert steps("reverse", 0) == steps("forward", 0) == steps("landing") == 16
    assert steps("delayed") == 4 * 16


def test_segment_reversed_distance_formula():
    """d_min = L - g + 1 + g * ceil((s - g + 1) / g)^+，g=1 时即 L + s。"""
    import math

    for size in (8, 16):
        for g in (x for x in (1, 2, 4, 8) if size % x == 0):
            for s in range(size + 1):
                d = dropped_distances(visibility("reverse", 3, 5, size, s, g))
                if d.numel():
                    k = max(0, math.ceil((s - g + 1) / g))
                    assert d.min() == size - g + 1 + g * k, (size, g, s)
