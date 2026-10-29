"""新旧权重按架构加载，旧模型不能被静默当成新版续训。"""

from argparse import Namespace
from dataclasses import asdict

import pytest
import torch

from latent_ar.checkpoints import (
    BLOCK_ARCHITECTURE,
    LEGACY_ARCHITECTURE,
    checkpoint_architecture,
    load_model,
)
from latent_ar.legacy_model import LatentARModel as LegacyModel
from latent_ar.legacy_model import ModelConfig as LegacyConfig
from latent_ar.model import LatentARModel, ModelConfig
from train import load_configuration


def dimensions():
    return dict(
        vocab_size=32,
        encoder_dim=8,
        dim=16,
        layers=1,
        heads=2,
        ff_dim=32,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ff_dim=32,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_loader_preserves_each_architecture(legacy):
    source = (
        LegacyModel(LegacyConfig(**dimensions()))
        if legacy
        else LatentARModel(ModelConfig(**dimensions()))
    )
    checkpoint = {"model_config": asdict(source.cfg), "model": source.state_dict()}
    if not legacy:
        checkpoint["architecture"] = BLOCK_ARCHITECTURE
    expected_architecture = LEGACY_ARCHITECTURE if legacy else BLOCK_ARCHITECTURE
    assert checkpoint_architecture(checkpoint) == expected_architecture
    restored = load_model(checkpoint, torch.device("cpu"))
    assert type(restored) is type(source)
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])


def test_training_rejects_legacy_checkpoint_before_touching_run(tmp_path):
    source = LegacyModel(LegacyConfig(**dimensions()))
    checkpoint = {"model_config": asdict(source.cfg), "model": source.state_dict()}
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    args = Namespace(resume=path)
    with pytest.raises(ValueError, match="旧版 AR"):
        load_configuration(args)
