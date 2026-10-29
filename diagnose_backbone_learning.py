"""比较已训练主干、恢复初始主干和抹去上下文，检查有效条件学习。"""

import argparse
import json
from pathlib import Path

import torch

from download_data import sha256
from latent_ar.checkpoints import load_model
from latent_ar.data import StoryDataset
from latent_ar.staged_training import evaluate_stage
from train import precision_context, select_device


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.initial, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if source["dataset_hash"] != checkpoint["dataset_hash"] or checkpoint[
        "dataset_hash"
    ] != sha256(Path("data/prepared/metadata.json")):
        raise ValueError("Dataset mismatch")
    if any(
        not torch.equal(value, checkpoint["model"][key])
        for key, value in source["model"].items()
        if key.startswith("codec.")
    ):
        raise ValueError("Codec weights differ")
    device = select_device("auto")
    model = load_model(checkpoint, device).eval()
    trained = {
        name: value.detach().clone()
        for name, value in model.backbone.state_dict().items()
    }
    initial = {
        name[len("backbone.") :]: value
        for name, value in source["model"].items()
        if name.startswith("backbone.")
    }
    delta = sum(
        float((trained[key].cpu().float() - value.float()).square().sum())
        for key, value in initial.items()
    )
    norm = sum(float(value.float().square().sum()) for value in initial.values())
    dataset = StoryDataset(Path("data/prepared"), "dev", 16, 128, 64, model.cfg.pad_id)
    autocast = precision_context(device, "bf16")

    def measure():
        return evaluate_stage("prior", model, dataset, 128, 1, 10042, device, autocast)

    results = {"trained": measure()}
    model.backbone.load_state_dict(initial, strict=True)
    results["restored_initial_backbone"] = measure()
    model.backbone.load_state_dict(trained, strict=True)
    hook = model.backbone.register_forward_hook(
        lambda module, inputs, output: (torch.zeros_like(output[0]), output[1])
    )
    try:
        results["zero_context"] = measure()
    finally:
        hook.remove()
    report = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "initial": str(args.initial),
        "initial_step": source["step"],
        "relative_backbone_parameter_change": (delta / norm) ** 0.5,
        "metrics": results,
        "notes": [
            "Same frozen codec, local output head, dev windows and evaluation protocol in every arm.",
            "Restoring initial weights diagnoses reliance and coadaptation; it is not a matched frozen-backbone training control.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
