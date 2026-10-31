from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from download_data import sha256
from experiments.distill_gru_feedback import (
    collect_split,
    student_metrics,
    train,
    truncate,
)
from latent_ar.feedback_student import FeedbackStudent, StudentConfig


class TinyDataset:
    size = 3

    def evaluation_batches(self, batch_size, max_batches, seed):
        del seed
        for start in range(0, min(self.size, batch_size * max_batches), batch_size):
            count = min(batch_size, self.size - start)
            yield {
                "prompt": torch.ones(count, 2, dtype=torch.long),
                "prompt_lengths": torch.full((count,), 2),
                "labels": torch.zeros(count, 1, dtype=torch.long),
            }


class TinyTeacher:
    cfg = SimpleNamespace(chunk_size=16, eos_id=2)

    def generate(self, prompt, prompt_lengths, **kwargs):
        del prompt_lengths, kwargs
        n = len(prompt)
        records = []
        for block in range(4):
            words = torch.full((n, 16), 3)
            if block == 1:
                words[0, 2] = 2
            records.append(
                {
                    "active": torch.ones(n, dtype=torch.bool),
                    "lengths": torch.full((n,), 16),
                    "token_ids": words,
                    "context": torch.full((n, 4), float(block)),
                    "feedback": torch.full((n, 4), float(block + 1)),
                }
            )
        return {"records": records}


def test_collect_filters_eos_last_block_and_records_story_ids():
    result = collect_split(
        TinyTeacher(), TinyDataset(), 3, 2, torch.device("cpu"), nullcontext, 7
    )
    assert result["h"].shape == (7, 4)
    assert result["block_index"].tolist().count(3) == 0
    assert result["story_id"].tolist() == [0, 1, 1, 0, 1, 2, 2]
    assert result["prompt_story_ids"].tolist() == [0, 1, 2]
    assert result["h"].dtype == result["r"].dtype == torch.float32


def test_metrics_and_eos_truncation():
    student = FeedbackStudent(
        StudentConfig(context_dim=4, feedback_dim=4, hidden_dim=4, layers=1)
    )
    data = {"h": torch.randn(5, 4), "r": torch.randn(5, 4)}
    stats = student_metrics(student, data, data["r"].mean(0), "cpu", 2)
    assert set(stats) == {"mse", "cosine", "train_mean_baseline_mse"}
    assert stats["mse"] >= 0 and stats["train_mean_baseline_mse"] >= 0
    assert truncate(torch.tensor([4, 2, 5]), 2) == [4, 2]


def make_data(path):
    path.mkdir()
    (path / "teacher.pt").write_bytes(b"teacher")
    (path / "tokenizer.json").write_bytes(b"tokenizer")
    for split in ("train", "dev"):
        torch.save(
            {
                "h": torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32,
                "r": torch.arange(32, dtype=torch.float32).reshape(8, 4) / 16,
            },
            path / f"{split}.pt",
        )
    manifest = {
        "teacher_snapshot_sha256": sha256(path / "teacher.pt"),
        "tokenizer_sha256": sha256(path / "tokenizer.json"),
        "split_sha256": {s: sha256(path / f"{s}.pt") for s in ("train", "dev")},
        "dataset_hash": "example",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_resume_matches_uninterrupted_training(tmp_path):
    data = tmp_path / "data"
    make_data(data)

    def args(run_dir, steps, resume=None):
        return SimpleNamespace(
            data=data,
            run_dir=run_dir,
            steps=steps,
            batch_size=4,
            device="cpu",
            seed=17,
            resume=resume,
        )

    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    train(args(continuous, 2))
    train(args(resumed, 1))
    train(args(resumed, 2, resumed / "last.pt"))
    a = torch.load(continuous / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 2
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)


def test_resume_rejects_different_batch_size(tmp_path):
    data = tmp_path / "data"
    make_data(data)
    run = tmp_path / "run"
    args = SimpleNamespace(
        data=data,
        run_dir=run,
        steps=1,
        batch_size=4,
        device="cpu",
        seed=17,
        resume=None,
    )
    train(args)
    args.steps, args.batch_size, args.resume = 2, 5, run / "last.pt"
    with pytest.raises(ValueError, match="批量"):
        train(args)


def test_attention_training_resume_and_config_validation(tmp_path):
    data = tmp_path / "data"
    make_data(data)
    for split in ("train", "dev"):
        path = data / f"{split}.pt"
        values = torch.load(path, weights_only=True)
        values["story_id"] = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        values["block_index"] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        torch.save(values, path)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_sha256"] = {s: sha256(data / f"{s}.pt") for s in ("train", "dev")}
    manifest_path.write_text(json.dumps(manifest))
    run = tmp_path / "run"
    args = SimpleNamespace(
        data=data,
        run_dir=run,
        steps=1,
        batch_size=4,
        device="cpu",
        seed=17,
        resume=None,
        student_type="attention",
        history_blocks=3,
    )
    train(args)
    saved = torch.load(run / "last.pt", weights_only=True)
    assert saved["architecture"] == "feedback_student_attention_v1"
    args.resume, args.steps = run / "last.pt", 2
    train(args)
    args.steps, args.history_blocks = 3, 2
    with pytest.raises(ValueError, match="结构"):
        train(args)


def test_add_prompt_history_preserves_source_and_verifies_output(tmp_path, monkeypatch):
    from experiments import distill_gru_feedback as experiment
    from tests.test_feedback_student import teacher

    source = tmp_path / "source"
    make_data(source)
    torch.save({"config": {"training": {"precision": "fp32"}}}, source / "teacher.pt")
    for split in ("train", "dev"):
        path = source / f"{split}.pt"
        values = torch.load(path, weights_only=True)
        values["h"] = torch.randn(8, 16)
        values["r"] = torch.randn(8, 16)
        values["story_id"] = torch.tensor([4, 9, 4, 9, 4, 9, 4, 9])
        values["block_index"] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        values["prompts"] = [torch.tensor([1, 4, 5, 6]), torch.tensor([1, 4, 5, 6, 7])]
        values["prompt_story_ids"] = torch.tensor([9, 4])
        torch.save(values, path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["teacher_snapshot_sha256"] = sha256(source / "teacher.pt")
    manifest["split_sha256"] = {s: sha256(source / f"{s}.pt") for s in ("train", "dev")}
    manifest_path.write_text(json.dumps(manifest))
    before = {
        name: sha256(source / name)
        for name in ("teacher.pt", "train.pt", "dev.pt", "manifest.json")
    }
    monkeypatch.setattr(
        experiment, "load_teacher", lambda data, device: (teacher().eval(), manifest)
    )
    output = tmp_path / "output"
    experiment.add_prompt_history(
        SimpleNamespace(
            data=source, output=output, device="cpu", history_blocks=2, batch_size=2
        )
    )
    assert before == {name: sha256(source / name) for name in before}
    result_manifest = experiment.verified_data(output)
    assert (
        result_manifest["prompt_history"]["source_split_sha256"]
        == manifest["split_sha256"]
    )
    for split in ("train", "dev"):
        original = torch.load(source / f"{split}.pt", weights_only=True)
        result = torch.load(output / f"{split}.pt", weights_only=True)
        for key in ("h", "r", "story_id", "block_index", "prompt_story_ids"):
            torch.testing.assert_close(result[key], original[key])
        assert all(
            torch.equal(a, b) for a, b in zip(result["prompts"], original["prompts"])
        )
        assert result["prefix_h"].shape == (2, 2, 16)
    with pytest.raises(FileExistsError):
        experiment.add_prompt_history(
            SimpleNamespace(
                data=source, output=output, device="cpu", history_blocks=2, batch_size=2
            )
        )
