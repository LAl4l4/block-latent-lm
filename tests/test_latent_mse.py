from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from latent_ar.staged import LatentPriorModel
from latent_ar.staged_training import configuration, loss_and_statistics
from test_staged import examples, small_config


def test_latent_only_objective_never_backpropagates_decoder_ce():
    torch.set_num_threads(1)
    model = LatentPriorModel(small_config())
    prompt, lengths, labels = examples()
    batch = dict(prompt=prompt, prompt_lengths=lengths, labels=labels)
    stats = loss_and_statistics("prior", model, batch, token_ce_grad=False)
    assert not stats["token_ce_sum"].requires_grad
    assert stats["latent_mse_sum"].requires_grad
    stats["latent_mse_sum"].backward()
    assert model.latent_head.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    assert all(p.grad is None for p in model.codec.parameters())


def test_latent_only_weights_are_allowed_but_empty_objective_is_rejected(tmp_path):
    config = json.loads(Path("paper/archive/configs/prior-latent-mse.json").read_text())
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = Namespace(
        resume=None,
        config=path,
        max_steps=None,
        batch_size=None,
        grad_accum=None,
        save_every=None,
        eval_batches=None,
        min_token_accuracy=None,
        min_block_accuracy=None,
    )
    result, _ = configuration(args, "prior")
    assert result["training"]["token_ce_weight"] == 0.0
    config["training"]["latent_mse_weight"] = 0.0
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="At least one"):
        configuration(args, "prior")
