import torch
import pytest

from experiments.feedback_study import noise_for, EvaluationStudent
from latent_ar.feedback_student import FeedbackStudent, StudentConfig


def test_noise_controls_match_capacity_and_blind_zero_input():
    student = FeedbackStudent(
        StudentConfig(
            context_dim=4, feedback_dim=4, hidden_dim=8, layers=1, noise_dim=3
        )
    )
    context = torch.randn(2, 4)
    noise = torch.rand(2, 3)
    assert torch.equal(noise_for("shared", noise), noise)
    assert torch.equal(noise_for("blind", noise), torch.full_like(noise, 0.5))
    assert noise_for("none", noise) is None
    assert torch.equal(
        EvaluationStudent(student, "blind")(context, noise),
        student(context, torch.full_like(noise, 0.5)),
    )


def test_resume_source_settings_are_checked(tmp_path, monkeypatch):
    from experiments import feedback_study
    from types import SimpleNamespace

    data = {
        "h": torch.ones(2, 4),
        "r": torch.ones(2, 4),
        "u": torch.ones(2, 3),
        "prompt_row": torch.arange(2),
        "prompts": [torch.ones(1), torch.ones(1)],
    }
    torch.save(data, tmp_path / "train.pt")
    torch.save(data, tmp_path / "dev.pt")
    manifest = {
        "token_temperature": 0,
        "split_sha256": {},
        "teacher_snapshot_sha256": "x",
    }
    monkeypatch.setattr(feedback_study, "verify", lambda _: manifest)
    resume = tmp_path / "resume.pt"
    torch.save({"study_settings": {}}, resume)
    args = SimpleNamespace(
        data=tmp_path,
        train_prompts=2,
        noise_mode="none",
        device="cpu",
        seed=1,
        hidden_dim=8,
        layers=1,
        batch_size=2,
        objective="mse",
        learning_rate=0.001,
        weight_decay=0.01,
        mse_weight=0.1,
        init_student=None,
        resume=resume,
        run_dir=tmp_path,
        steps=2,
    )
    with pytest.raises(ValueError, match="Resume settings"):
        feedback_study.train(args)


def test_matched_augmentation_preserves_story_mapping(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import feedback_study as study

    def make_data(ids, stories):
        rows = torch.tensor(stories, dtype=torch.long)
        data = {
            key: torch.arange(len(rows), dtype=torch.float32)[:, None]
            for key in ("h", "r", "u")
        }
        data.update(
            story_id=torch.tensor([ids[i] for i in stories]),
            block_index=torch.zeros(len(rows), dtype=torch.long),
            prompt_row=rows,
            past_feedback=torch.zeros(len(rows), 3, 1),
            past_lengths=torch.zeros(len(rows), dtype=torch.long),
            next_tokens=torch.ones(len(rows), 2, dtype=torch.long),
            prompts=[torch.tensor([i]) for i in ids],
            prompt_story_ids=torch.tensor(ids),
        )
        return data

    base_dir, student_dir = tmp_path / "base", tmp_path / "student"
    for directory in (base_dir, student_dir):
        directory.mkdir()
        for name in ("teacher.pt", "tokenizer.json", "dev.pt", "test.pt"):
            (directory / name).write_bytes(name.encode())
    base = make_data([0, 1, 2, 3], [0, 0, 1, 2, 2, 2, 3])
    student = make_data([2, 3], [0, 1])
    student["next_tokens"].fill_(-100)
    torch.save(base, base_dir / "train.pt")
    torch.save(student, student_dir / "train.pt")
    common = {
        "teacher_snapshot_sha256": "teacher",
        "tokenizer_sha256": "tokenizer",
        "dataset_hash": "dataset",
        "token_temperature": 0,
        "counts": {},
        "split_sha256": {"train": "basehash", "dev": "devhash", "test": "testhash"},
    }
    student_manifest = dict(common) | {
        "on_policy": {
            "source_train_sha256": "basehash",
            "student_sha256": "studenthash",
        },
        "split_sha256": {"train": "studenthash", "dev": "devhash", "test": "testhash"},
    }
    monkeypatch.setattr(
        study, "verify", lambda path: common if path == base_dir else student_manifest
    )
    off, on = tmp_path / "off", tmp_path / "on"
    study.matched_augment(
        SimpleNamespace(
            base_data=base_dir,
            student_data=student_dir,
            output_offpolicy=off,
            output_onpolicy=on,
            base_prompts=2,
            seed=2034,
        )
    )
    a, b = (torch.load(path / "train.pt", weights_only=True) for path in (off, on))
    assert len(a["h"]) == len(b["h"]) == 5
    assert torch.equal(a["h"][:3], b["h"][:3])
    for data in (a, b):
        assert torch.equal(
            data["prompt_story_ids"][data["prompt_row"]], data["story_id"]
        )
        assert data["prompt_story_ids"].tolist() == [0, 1, 2, 3]
    assert (b["next_tokens"] == -100).all()


def test_student_history_records_predicted_feedback(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from experiments.feedback_study import collect_split, PromptDataset
    import latent_ar.feedback_student as module

    def fake_generate(teacher, student, padded, lengths, **kwargs):
        records = []
        for i in range(3):
            records.append(
                {
                    "active": torch.tensor([True]),
                    "lengths": torch.tensor([2]),
                    "token_ids": torch.tensor([[3, 4]]),
                    "context": torch.tensor([[float(i)]]),
                    "noise": torch.zeros(1, 2),
                    "true_feedback": torch.tensor([[10.0 + i]]),
                    "predicted_feedback": torch.tensor([[20.0 + i]]),
                }
            )
        return {"records": records}

    monkeypatch.setattr(module, "generate_student_feedback", fake_generate)
    teacher = SimpleNamespace(
        cfg=SimpleNamespace(pad_id=0, eos_id=2, chunk_size=2, dim=1)
    )
    values = collect_split(
        teacher,
        PromptDataset([torch.tensor([5])]),
        0,
        1,
        1,
        torch.device("cpu"),
        nullcontext,
        2031,
        student=object(),
    )
    assert values["r"].flatten().tolist() == [10.0, 11.0]
    assert values["past_feedback"][1, 0, 0].item() == 20.0
    assert (values["next_tokens"] == -100).all()


def test_rollout_seed_is_independent_of_prompt_seed(monkeypatch):
    import random
    from contextlib import nullcontext
    from types import SimpleNamespace
    from experiments.feedback_study import collect_split
    import latent_ar.feedback_student as module

    seen_seeds = []

    class Dataset:
        size = 1

        def example(self, index, rng):
            return torch.tensor([rng.randrange(100000)]), None

    def fake_generate(teacher, student, padded, lengths, generator, **kwargs):
        seen_seeds.append(generator.initial_seed())
        record = {
            "active": torch.tensor([True]),
            "lengths": torch.tensor([2]),
            "token_ids": torch.tensor([[3, 4]]),
            "context": torch.zeros(1, 1),
            "noise": torch.rand(1, 2, generator=generator),
            "true_feedback": torch.zeros(1, 1),
            "predicted_feedback": torch.zeros(1, 1),
        }
        return {"records": [record, record]}

    monkeypatch.setattr(module, "generate_student_feedback", fake_generate)
    teacher = SimpleNamespace(
        cfg=SimpleNamespace(pad_id=0, eos_id=2, chunk_size=2, dim=1)
    )
    outputs = [
        collect_split(
            teacher,
            Dataset(),
            0,
            1,
            1,
            torch.device("cpu"),
            nullcontext,
            2031,
            student=object(),
            rollout_seed=seed,
        )
        for seed in (2031, 1002031, 2002031)
    ]
    assert seen_seeds == [2031, 1002031, 2002031]
    assert all(
        int(out["prompts"][0][0]) == random.Random(2031).randrange(100000)
        for out in outputs
    )
    assert not torch.equal(outputs[0]["u"], outputs[1]["u"])
    assert not torch.equal(outputs[1]["u"], outputs[2]["u"])


def test_teacher_precision_uses_snapshot_and_reports_cpu_actual(monkeypatch, tmp_path):
    from experiments import feedback_study as study

    monkeypatch.setattr(
        study.torch,
        "load",
        lambda *args, **kwargs: {"config": {"training": {"precision": "bf16"}}},
    )
    seen = []
    monkeypatch.setattr(
        study,
        "precision_context",
        lambda device, precision: seen.append((device.type, precision)) or object(),
    )
    configured, actual, factory = study.teacher_precision(tmp_path, torch.device("cpu"))
    assert (configured, actual) == ("bf16", "fp32")
    assert seen == [("cpu", "bf16")]
    assert factory is not None
