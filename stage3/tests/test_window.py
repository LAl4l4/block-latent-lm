import torch

from stage3.models import WindowLM
from stage2.schedules import visibility
from stage3.schedules import window_visibility
from stage3.window.train import losses

P, B, L, D = 4, 4, 4, 3
ARMS = [
    ("zonly", None, 1),
    ("causal", None, 1),
    ("forward", 2, 1),
    ("reverse", 2, 1),
    ("reverse", 2, 2),
]


def test_window_only_removes_far_blocks():
    n = P + B * L
    block = torch.where(torch.arange(n) >= P, (torch.arange(n) - P) // L, -1)
    for schedule, stagger, segment in ARMS[1:]:
        base = visibility(schedule, P, B, L, stagger, segment)
        for window in (0, 1, 2):
            v = window_visibility(schedule, P, B, L, stagger, segment, window)
            assert not (v & ~base).any()
            far = (block[:, None] - block[None, :] > window) & (block[:, None] >= 0)
            assert not v[far].any()
            assert torch.equal(v[~far], base[~far])
    zonly = window_visibility("zonly", P, B, L)
    assert not zonly[P:].any()


def test_predictions_depend_exactly_on_visible_words_and_own_latent():
    """float64 下逐个改动词或 z：不可见输入（含别的块的 z）的影响必须精确为 0。

    一层时只读窗口内的词；多层时窗口经内容流逐层外推，但读到的仍只是更早生成的词。
    """
    for layers in (1, 2):
        for gate in (False, True):
            check_dependencies(layers, gate)


def check_dependencies(layers, gate):
    torch.manual_seed(0)
    n = P + B * L
    model = WindowLM(50, n, B, L, D, 32, layers, 2, 64, gate=gate).double().eval()
    if gate:
        # α 初始为 0 时左边那一路不起作用；打开它，测试才能覆盖这一路的可见关系。
        for g in model.left_layers:
            g.alpha.data.fill_(0.5)
    ids = torch.randint(0, 50, (1, n))
    fixed = torch.randn(1, B, D, dtype=torch.float64)
    words = ids.repeat(n + 1, 1)
    for position in range(n):
        words[position + 1, position] = (words[position + 1, position] + 1) % 50
    z = fixed.repeat(B + 1, 1, 1)
    for c in range(B):
        z[c + 1, c] += 1.0
    target_block = torch.arange(B * L) // L
    for schedule, stagger, segment in ARMS:
        v = window_visibility(schedule, P, B, L, stagger, segment, window=1)
        legal = v if layers == 1 else window_visibility(
            schedule, P, B, L, stagger, segment, window=B)
        with torch.no_grad():
            out = model(words, v, P, fixed.expand(n + 1, -1, -1))
            change = (out[1:] - out[:1]).abs().amax(-1)
            assert change[~legal[P:].T].max() == 0, schedule
            out = model(ids.expand(B + 1, -1), v, P, z)
            change = (out[1:] - out[:1]).abs().amax(-1)
            other = target_block[None, :] != torch.arange(B)[:, None]
            assert change[other].max() == 0, schedule
            assert (change[~other] > 0).all(), schedule


def test_posterior_reads_only_its_own_block_and_losses_backprop():
    torch.manual_seed(0)
    model = WindowLM(50, P + B * L, B, L, D, 32, 2, 2, 64)
    ids = torch.randint(0, 50, (2, P + B * L))
    changed = ids.clone()
    changed[:, P + 2 * L + 1] = (changed[:, P + 2 * L + 1] + 1) % 50
    with torch.no_grad():
        mean, _ = model.encode(ids, P)
        mean2, _ = model.encode(changed, P)
    same = (mean - mean2).abs().amax(-1)[0]
    assert (same[[0, 1, 3]] == 0).all() and same[2] > 0
    word, kl = losses(model, ids, window_visibility("reverse", P, B, L, 2), P)
    assert word.shape == (2, B, L) and kl.shape == (2, B, D) and (kl >= -1e-6).all()
    (word.mean() + kl.mean()).backward()
    assert model.posterior.weight.grad.abs().sum() > 0


def test_sampling_never_reads_unwritten_positions():
    """占位值不同、随机种子相同时，采出的词必须完全一致。"""
    from stage3.window.samples import sample

    torch.manual_seed(0)
    model = WindowLM(50, P + B * L, B, L, D, 32, 2, 2, 64).double().eval()
    prompt_ids = torch.randint(0, 50, (2, P))
    z = torch.randn(2, B, D, dtype=torch.float64)
    for schedule, stagger, segment in ARMS:
        v = window_visibility(schedule, P, B, L, stagger, segment, window=1)
        out = [sample(model, v, prompt_ids, P, z, 1.0, torch.Generator().manual_seed(1), fill)
               for fill in (0, 7)]
        assert torch.equal(out[0], out[1]), schedule
        assert torch.equal(out[0][:, :P], prompt_ids)


def test_gated_model_reads_left_blocks_only_through_the_gate():
    """α 为 0 时，左边块的词对预测的影响精确为 0；打开后才有影响。"""
    torch.manual_seed(0)
    n = P + B * L
    model = WindowLM(50, n, B, L, D, 32, 2, 2, 64, gate=True).double().eval()
    v = window_visibility("causal", P, B, L, window=1)
    ids = torch.randint(0, 50, (1, n)).repeat(2, 1)
    ids[1, P + L - 1] = (ids[1, P + L - 1] + 1) % 50  # 改第 0 块最后一个词
    z = torch.randn(1, B, D, dtype=torch.float64).expand(2, -1, -1)
    with torch.no_grad():
        out = model(ids, v, P, z)
        assert (out[0, L:2 * L] - out[1, L:2 * L]).abs().max() == 0
        for g in model.left_layers:
            g.alpha.data.fill_(0.5)
        out = model(ids, v, P, z)
        assert (out[0, L:2 * L] - out[1, L:2 * L]).abs().max() > 0


def test_corrupt_keeps_prompt_and_shapes():
    from stage3.window.train import corrupt

    torch.manual_seed(0)
    ids = torch.randint(0, 50, (8, P + B * L))
    z = torch.randn(8, B, D)
    inputs, noisy = corrupt(ids, z, P, L, 50)
    assert inputs.shape == ids.shape and noisy.shape == z.shape
    assert torch.equal(inputs[:, :P], ids[:, :P])
    assert (inputs[:, P:] != ids[:, P:]).any() and inputs.max() < 50
