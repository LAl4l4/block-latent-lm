"""调度可见关系：因果基线、无泄漏、传递性，以及倒序的距离保证。"""

import torch

from seam.models import TwoStreamLM
from seam.schedules import visibility

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


def test_reverse_matches_hand_drawn_example():
    """2 块、L=4、s=2、prompt=1，逐格对照手画矩阵（0=prompt，1..4 第 0 块，5..8 第 1 块）。"""
    reads = {
        1: [0, 2, 3, 4], 2: [0, 3, 4], 3: [0, 4], 4: [0],
        5: [0, 1, 2, 3, 4, 6, 7, 8], 6: [0, 1, 2, 3, 4, 7, 8],
        7: [0, 2, 3, 4, 8], 8: [0, 3, 4],
    }
    expected = torch.zeros(9, 9, dtype=torch.bool)
    for i, keys in reads.items():
        expected[i, keys] = True
    assert torch.equal(visibility("reverse", 1, 2, 4, 2), expected)


def simulated_visibility(schedule, prompt, blocks, size, stagger=None, segment=1):
    """不用时钟公式：逐块列出写入顺序，按"更早写完且不在右侧块"重算可见关系。"""
    clock = {}
    for b in range(blocks):
        if schedule == "causal":
            order, start = list(range(size)), b * size
        elif schedule == "forward":
            order, start = list(range(size)), b * stagger
        elif schedule in ("meet", "alternate"):
            order = list(range(size)) if b % 2 == 0 else list(range(size - 1, -1, -1))
            if schedule == "meet":
                start = (b // 2) * (size + stagger) + (b % 2) * stagger
            else:
                start = b * stagger
        else:
            segments = [list(range(j, j + segment)) for j in range(0, size, segment)]
            order, start = [x for seg in reversed(segments) for x in seg], b * stagger
        for t, slot in enumerate(order):
            clock[prompt + b * size + slot] = (start + t, b)
    n = prompt + blocks * size
    v = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for k in range(n):
            if i < prompt:
                v[i, k] = k < i
            elif k < prompt:
                v[i, k] = True
            else:
                v[i, k] = clock[k][0] < clock[i][0] and clock[k][1] <= clock[i][1]
    return v


def test_visibility_matches_simulated_write_order():
    for size in (4, 8):
        assert torch.equal(visibility("causal", 3, 3, size), simulated_visibility("causal", 3, 3, size))
        for s in range(size + 1):
            assert torch.equal(
                visibility("forward", 3, 3, size, s), simulated_visibility("forward", 3, 3, size, s)
            )
            for g in (x for x in (1, 2, 4) if size % x == 0):
                assert torch.equal(
                    visibility("reverse", 3, 3, size, s, g),
                    simulated_visibility("reverse", 3, 3, size, s, g),
                ), (size, s, g)


def test_schedules_are_transitive_and_acyclic():
    for schedule, stagger, segment in [
        ("causal", None, 1), ("delayed", None, 1), ("forward", 4, 1), ("reverse", 4, 1),
        ("reverse", 1, 1), ("forward", 8, 1), ("reverse", 8, 1),
        ("reverse", 0, 1), ("forward", 0, 1), ("landing", None, 1),
        ("reverse", 4, 4), ("reverse", 4, 2), ("reverse", 0, 4),
        ("meet", 0, 1), ("meet", 1, 1), ("meet", 4, 1), ("alternate", 1, 1), ("alternate", 0, 1),
    ]:
        v = visibility(schedule, PROMPT, BLOCKS, L, stagger, segment).int()
        assert not torch.diagonal(v).any()
        assert not (v & v.T).any()
        # 读到的位置所读到的一切，自己也能读到：查询流不会从内容流间接泄漏。
        assert ((v @ v) > 0).le(v.bool()).all(), (schedule, stagger, segment)


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


def test_prediction_depends_exactly_on_visible_tokens():
    """论文实际配置（prompt 16、7 块、L=16）：逐个改动输入词，float64 下不可见词的影响
    必须精确为 0，可见词必须改变输出（检验本测试有发现泄漏的能力）。"""
    torch.manual_seed(0)
    prompt, blocks, size = 16, 7, 16
    n = prompt + blocks * size
    model = TwoStreamLM(64, n, 32, 3, 4, 64).double().eval()
    targets = torch.arange(prompt, n)
    batch = torch.randint(0, 64, (1, n)).repeat(n + 1, 1)
    for position in range(n):
        batch[position + 1, position] = (batch[position + 1, position] + 1) % 64
    for schedule, stagger, segment in [
        ("causal", None, 1), ("delayed", None, 1), ("forward", 8, 1), ("reverse", 8, 1),
        ("reverse", 8, 8), ("reverse", 0, 1), ("reverse", 0, 8), ("landing", None, 1),
    ]:
        v = visibility(schedule, prompt, blocks, size, stagger, segment)
        with torch.no_grad():
            out = model(batch, v, targets)
        change = (out[1:] - out[:1]).abs().amax(-1)  # [改动的位置, 预测目标]
        seen = v[targets].T
        assert change[~seen].max() == 0, (schedule, stagger, segment)
        assert (change[seen] > 0).all(), (schedule, stagger, segment)


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
    from seam.schedules import rounds

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


def test_meet_matches_simulated_write_order():
    for size in (4, 8):
        for d in range(size + 1):
            for schedule in ("meet", "alternate"):
                assert torch.equal(
                    visibility(schedule, 3, 4, size, d),
                    simulated_visibility(schedule, 3, 4, size, d),
                ), (schedule, size, d)


def test_meet_seam_and_distance():
    """组内倒序块晚 d 步：其块首读到前一块块尾，丢失词对的最小距离是 d+1。"""
    for d in range(1, L):
        v = visibility("meet", PROMPT, BLOCKS, L, d)
        head = PROMPT + L  # 第 1 块（倒序）的块首
        assert v[head, PROMPT + L - 1]
        assert int(dropped_distances(v).min()) == d + 1
    # 组与组之间不丢任何依赖：第 2 块能读到第 1 块全部。
    v = visibility("meet", PROMPT, BLOCKS, L, 1)
    assert v[PROMPT + 2 * L, PROMPT + L : PROMPT + 2 * L].all()
