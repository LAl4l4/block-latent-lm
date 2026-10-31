import torch
import torch.nn.functional as F

from seam.models import PlanLM
from seam.schedules import plan_visibility, visibility
from seam.train_plan import losses

P, B, L, K = 4, 4, 8, 2
ARMS = [
    ("causal", None, 1),
    ("forward", 4, 1),
    ("reverse", 4, 1),
    ("reverse", 4, 4),
    ("reverse", 8, 4),
    ("reverse", 0, 1),
    ("meet", 1, 1),
    ("meet", 3, 1),
]


def build(schedule, stagger, segment):
    return plan_visibility(schedule, P, B, L, K, stagger, segment)


def test_word_part_matches_mask_study():
    n = P + B * L
    for schedule, stagger, segment in ARMS:
        v = build(schedule, stagger, segment)
        assert torch.equal(v[:n, :n], visibility(schedule, P, B, L, stagger, segment))


def test_acyclic_and_transitive():
    for arm in ARMS:
        v = build(*arm)
        assert not (v & v.T).any()
        assert not v.diagonal().any()
        reach = (v.float() @ v.float()) > 0
        assert not (reach & ~v).any()


def test_causal_plan_sees_previous_blocks_only():
    v = build("causal", None, 1)
    n = P + B * L
    for t in range(B):
        row = v[n + t * K]
        assert row[: P + t * L].all()
        assert not row[P + t * L : n].any()


def test_staggered_plan_sees_words_written_before_block_start():
    # reverse s=4 g=4：第 t 块起步时，第 t-1 块只写完了后一段（槽 4..7）。
    v = build("reverse", 4, 4)
    n = P + B * L
    for t in range(1, B):
        row = v[n + t * K]
        previous = row[P + (t - 1) * L : P + t * L]
        assert previous.tolist() == [False] * 4 + [True] * 4


def test_words_see_own_and_earlier_plans_only():
    n = P + B * L
    for arm in ARMS:
        v = build(*arm)
        for i in range(P, n):
            b = (i - P) // L
            plans = v[i, n:].view(B, K)
            assert plans[: b + 1].all() and not plans[b + 1 :].any()
        assert not v[:P, n:].any()


def test_predictions_ignore_invisible_inputs():
    torch.manual_seed(0)
    model = PlanLM(50, P + B * L, B, L, K, 5, 32, 2, 2, 64, encoder_dim=8).eval()
    v = build("reverse", 4, 4)
    ids = torch.randint(0, 50, (1, P + B * L))
    words, plans, _, _ = model(ids, v, P)
    changed = ids.clone()
    changed[0, P + 2 * L + 3] = (changed[0, P + 2 * L + 3] + 1) % 50
    words2, plans2, _, _ = model(changed, v, P)
    # 第 1 块的词看不到第 2 块的词和计划；第 2 块第一个码看不到第 2 块的词。
    assert torch.allclose(words[0, : 2 * L], words2[0, : 2 * L], atol=1e-5)
    assert torch.allclose(plans[0, 2 * K], plans2[0, 2 * K], atol=1e-5)


def test_losses_shapes_and_gradient_reaches_encoder():
    torch.manual_seed(0)
    model = PlanLM(50, P + B * L, B, L, K, 5, 32, 2, 2, 64, encoder_dim=8)
    ids = torch.randint(0, 50, (2, P + B * L))
    word, plan, index = losses(model, ids, build("causal", None, 1), P)
    assert word.shape == (2, B, L) and plan.shape == (2, B, K)
    assert index.shape == (2, B * K)
    (word.mean() + plan.mean()).backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_predictions_depend_exactly_on_visible_words_and_codes():
    """码固定（同采样时），逐个改动词或码：float64 下不可见输入的影响必须精确为 0。"""
    torch.manual_seed(0)
    n, choices = P + B * L, 5
    model = PlanLM(50, n, B, L, K, choices, 32, 2, 2, 64, encoder_dim=8).double().eval()
    rows = torch.arange(P, n + B * K)
    ids = torch.randint(0, 50, (1, n))
    fixed = torch.randint(0, choices, (1, B * K))

    def run(words, codes):
        model.encode = lambda ids_, prompt: (F.one_hot(codes, choices).double(), codes)
        with torch.no_grad():
            word_logits, plan_logits, _, _ = model(words, v, P)
        return torch.cat((
            (word_logits[1:] - word_logits[:1]).abs().amax(-1),
            (plan_logits[1:] - plan_logits[:1]).abs().amax(-1),
        ), 1)  # [改动的输入, 输出行]

    words = ids.repeat(n + 1, 1)
    for position in range(n):
        words[position + 1, position] = (words[position + 1, position] + 1) % 50
    codes = fixed.repeat(B * K + 1, 1)
    for c in range(B * K):
        codes[c + 1, c] = (codes[c + 1, c] + 1) % choices
    for arm in ARMS:
        v = build(*arm)
        seen = v[rows].T
        assert run(words, fixed.expand(n + 1, -1))[~seen[:n]].max() == 0, arm
        assert run(ids.expand(B * K + 1, -1), codes)[~seen[n:]].max() == 0, arm


def test_meet_plan_is_written_at_block_start():
    """meet 的第 1 块（倒序）晚 d 步起步：其计划码只能看到第 0 块最先写的 d 个词。"""
    for d in (1, 3):
        v = build("meet", d, 1)
        n = P + B * L
        row = v[n + 1 * K]
        assert row[P : P + d].all() and not row[P + d : P + L].any()
        assert not row[P + L : n].any()
