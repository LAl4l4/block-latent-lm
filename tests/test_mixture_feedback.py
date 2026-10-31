"""16-token 条件分布训练与实际文本回灌的集成检查。"""

from dataclasses import asdict

import torch

from latent_ar.checkpoints import load_model
from latent_ar.mixture import LatentMixtureModel, MixtureModelConfig


def small_model():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    return LatentMixtureModel(
        MixtureModelConfig(
            vocab_size=32,
            chunk_size=16,
            encoder_dim=16,
            dim=32,
            layers=1,
            heads=4,
            ff_dim=64,
            decoder_dim=16,
            decoder_layers=1,
            decoder_heads=2,
            decoder_ff_dim=32,
            mixture_components=4,
            mixture_hidden_dim=32,
            mixture_layers=1,
            mixture_train_blocks=2,
            feedback_mode="text",
        )
    )


def test_sixteen_token_distribution_gradient_and_no_target_leak():
    model = small_model()
    prompt = torch.randint(4, 32, (2, 19))
    lengths = torch.tensor([19, 16])
    labels = torch.randint(4, 32, (2, 32))
    labels[1, 21:] = -100
    before, _, _ = model.teacher_contexts(prompt, lengths, labels)
    changed = labels.clone()
    changed[:, :16] = 25
    after, _, _ = model.teacher_contexts(prompt, lengths, changed)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])

    stats = model.prior_statistics(prompt, lengths, labels)
    assert stats["blocks"] == 4 and stats["scored_blocks"] == 2
    loss = stats["text_nll_sum"] / stats["tokens"]
    assert torch.isfinite(loss)
    loss.backward()
    for module in [model.backbone, model.latent_head]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert all(not p.requires_grad and p.grad is None for p in model.codec.parameters())
    assert not model.codec.training


def test_text_feedback_survives_checkpoint_and_matches_emitted_blocks():
    model = small_model()
    prompt = torch.randint(4, 32, (2, 19))
    lengths = torch.tensor([19, 16])

    def sample(current, mode):
        return current.generate(
            prompt, lengths,
            max_new_tokens=35,
            min_new_tokens=35,
            generator=torch.Generator().manual_seed(91),
            feedback_mode=mode,
            return_trace=True,
        )

    text = sample(model, "text")
    raw = sample(model, "raw")
    assert model.training  # 推理应恢复调用前状态，codec 始终冻结。
    assert text["output_ids"].shape == (2, 35)
    assert len(text["records"]) == 3
    torch.testing.assert_close(text["output_ids"][:, :16], raw["output_ids"][:, :16])
    for record in text["records"]:
        encoded, _ = model.codec.encode(record["token_ids"], record["lengths"])
        torch.testing.assert_close(record["feedback"], encoded[:, 0])
        assert not record["z"].requires_grad
    assert text["records"][-1]["lengths"].tolist() == [3, 3]

    restored = load_model({
        "architecture": model.architecture,
        "model_config": asdict(model.cfg),
        "model": model.state_dict(),
    }, torch.device("cpu"))
    assert restored.cfg.feedback_mode == "text"
    torch.testing.assert_close(text["output_ids"], sample(restored, "text")["output_ids"])


def test_old_mixture_checkpoint_keeps_raw_feedback_default():
    model = small_model()
    config = asdict(model.cfg)
    del config["feedback_mode"]
    restored = load_model({
        "architecture": model.architecture,
        "model_config": config,
        "model": model.state_dict(),
    }, torch.device("cpu"))
    assert restored.cfg.feedback_mode == "raw"
