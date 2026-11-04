import torch

from stage3.models import LatentLM
from stage2.schedules import plan_visibility
from stage3.latent.train import gaussian_kl, losses

P, B, L, K, D = 4, 4, 8, 2, 3
ARMS = [
    ("causal", None, 1),
    ("forward", 4, 1),
    ("reverse", 4, 1),
    ("reverse", 4, 4),
    ("meet", 1, 1),
]


def build(schedule, stagger, segment):
    return plan_visibility(schedule, P, B, L, K, stagger, segment)


def test_gaussian_kl_matches_torch():
    torch.manual_seed(0)
    mq, lq, mp, lp = torch.randn(4, 5).unbind(0)
    q = torch.distributions.Normal(mq, (0.5 * lq).exp())
    p = torch.distributions.Normal(mp, (0.5 * lp).exp())
    expected = torch.distributions.kl_divergence(q, p)
    assert torch.allclose(gaussian_kl(mq, lq, mp, lp), expected, atol=1e-6)
    assert gaussian_kl(mq, lq, mq, lq).abs().max() < 1e-6


def test_losses_shapes_and_gradients():
    torch.manual_seed(0)
    model = LatentLM(50, P + B * L, B, L, K, D, 32, 2, 2, 64, encoder_dim=8)
    ids = torch.randint(0, 50, (2, P + B * L))
    word, kl = losses(model, ids, build("reverse", 4, 4), P)
    assert word.shape == (2, B, L) and kl.shape == (2, B, K, D)
    assert (kl >= -1e-6).all()
    (word.mean() + kl.mean()).backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.prior_output.weight.grad.abs().sum() > 0


def test_predictions_depend_exactly_on_visible_words_and_latents():
    """z 固定（同采样时），逐个改动词或潜变量：float64 下不可见输入的影响必须精确为 0。"""
    torch.manual_seed(0)
    n = P + B * L
    model = LatentLM(50, n, B, L, K, D, 32, 2, 2, 64, encoder_dim=8).double().eval()
    rows = torch.arange(P, n + B * K)
    ids = torch.randint(0, 50, (1, n))
    fixed = torch.randn(1, B * K, D, dtype=torch.float64)

    def run(words, z):
        with torch.no_grad():
            word_logits, mean_p, logvar_p = model(words, v, P, z)
        prior = torch.cat((mean_p, logvar_p), -1)
        return torch.cat((
            (word_logits[1:] - word_logits[:1]).abs().amax(-1),
            (prior[1:] - prior[:1]).abs().amax(-1),
        ), 1)  # [改动的输入, 输出行]

    words = ids.repeat(n + 1, 1)
    for position in range(n):
        words[position + 1, position] = (words[position + 1, position] + 1) % 50
    z = fixed.repeat(B * K + 1, 1, 1)
    for c in range(B * K):
        z[c + 1, c] += 1.0
    for arm in ARMS:
        v = build(*arm)
        seen = v[rows].T
        assert run(words, fixed.expand(n + 1, -1, -1))[~seen[:n]].max() == 0, arm
        assert run(ids.expand(B * K + 1, -1), z)[~seen[n:]].max() == 0, arm


def test_posterior_reads_only_its_own_block():
    torch.manual_seed(0)
    model = LatentLM(50, P + B * L, B, L, K, D, 32, 2, 2, 64, encoder_dim=8).eval()
    ids = torch.randint(0, 50, (1, P + B * L))
    changed = ids.clone()
    changed[0, P + 2 * L + 1] = (changed[0, P + 2 * L + 1] + 1) % 50
    mean, _ = model.encode(ids, P)
    mean2, _ = model.encode(changed, P)
    same = (mean - mean2).abs().amax(-1)[0].view(B, K)
    assert (same[[0, 1, 3]] == 0).all() and (same[2] > 0).any()


def test_sampling_never_reads_unwritten_positions():
    """占位值不同、随机种子相同时，采出的词和潜变量必须完全一致。"""
    from stage3.latent.samples import sample

    torch.manual_seed(0)
    model = LatentLM(50, P + B * L, B, L, K, D, 32, 2, 2, 64, encoder_dim=8).double().eval()
    prompt_ids = torch.randint(0, 50, (2, P))
    for schedule, stagger, segment in ARMS[:4]:
        c = {"schedule": schedule, "prompt": P, "blocks": B, "block_size": L,
             "latents": K, "latent_dim": D, "stagger": stagger, "segment": segment}
        v = build(schedule, stagger, segment)
        out = [
            sample(model, v, c, prompt_ids, 1.0, torch.Generator().manual_seed(1), fill)
            for fill in (0, 7)
        ]
        assert torch.equal(out[0][0], out[1][0]), schedule
        assert torch.allclose(out[0][1].double(), out[1][1].double()), schedule
        assert torch.equal(out[0][0][:, :P], prompt_ids)
