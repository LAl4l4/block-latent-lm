"""结构对照的文字联合分布、实际内容编码和回灌检查。"""

from dataclasses import asdict
import torch

from latent_ar.checkpoints import load_model
from latent_ar.token_head import LatentTokenHeadModel, TokenHeadConfig
from test_staged import small_config, examples


def test_rollout_feeds_chosen_latent_and_never_waits_for_text_decoder(monkeypatch):
    model = make_model()
    model.eval()
    prompt, lengths, _ = examples()
    inputs = []
    hook = model.backbone.register_forward_pre_hook(
        lambda _, args: inputs.append(args[0].detach().clone())
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Text decoder must not participate in latent rollout")

    monkeypatch.setattr(model.codec, "decode", forbidden)
    with torch.inference_mode():
        output = model.rollout(
            prompt, lengths, 3, generator=torch.Generator().manual_seed(42)
        )
    hook.remove()
    torch.testing.assert_close(inputs[1][:, 0], output[:, 0])
    torch.testing.assert_close(inputs[2][:, 0], output[:, 1])
    assert inputs[1].is_floating_point()


def make_model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return LatentTokenHeadModel(
        TokenHeadConfig(**asdict(small_config()), token_head_dim=16)
    )


def test_local_head_is_causal_and_has_gradients_to_backbone():
    model = make_model()
    contexts = torch.randn(2, 3, model.cfg.dim)
    previous = torch.randint(4, 32, (2, 3, 4))
    first = model.latent_head(contexts, previous)
    changed = previous.clone()
    changed[:, :, 2:] = 23
    second = model.latent_head(contexts, changed)
    torch.testing.assert_close(first[:, :, :2], second[:, :, :2])
    stats = model.prior_statistics(*examples())
    stats["text_nll_sum"].backward()
    assert model.backbone.blocks[0].attention.qkv.weight.grad.abs().sum() > 0
    assert model.latent_head.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.codec.parameters())


def test_sampled_latent_is_exact_encoding_of_selected_codes():
    model = make_model()
    context = torch.randn(2, model.cfg.dim)
    seen = []
    hook = model.codec.encoder.register_forward_pre_hook(
        lambda _, args: seen.append((args[0].clone(), args[1].clone()))
    )
    sampled = model.sample_latent(context, generator=torch.Generator().manual_seed(42))
    hook.remove()
    expected, _ = model.codec.encode(*seen[0])
    torch.testing.assert_close(sampled, expected[:, 0])


def test_token_head_checkpoint_and_seeded_rollout():
    model = make_model()
    restored = load_model(
        dict(
            architecture=model.architecture,
            model_config=asdict(model.cfg),
            model=model.state_dict(),
        ),
        torch.device("cpu"),
    )
    prompt, lengths, _ = examples()
    with torch.inference_mode():
        expected = model.rollout(
            prompt, lengths, 3, generator=torch.Generator().manual_seed(42)
        )
        actual = restored.rollout(
            prompt, lengths, 3, generator=torch.Generator().manual_seed(42)
        )
    torch.testing.assert_close(expected, actual)
