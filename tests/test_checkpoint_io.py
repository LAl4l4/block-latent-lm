from pathlib import Path

import pytest

from latent_ar.checkpoint_io import publish_checkpoint


def test_transient_windows_read_lock_is_retried(tmp_path, monkeypatch):
    previous = tmp_path / "last.pt"
    previous.write_bytes(b"previous")
    temporary = tmp_path / "last.tmp"
    temporary.write_bytes(b"complete new state")
    real_replace = Path.replace
    calls = []

    def locked(source, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError("simulated Windows read lock")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("latent_ar.checkpoint_io.time.sleep", lambda delay: None)
    assert publish_checkpoint(temporary, previous, 3100) == previous
    assert previous.read_bytes() == b"complete new state"
    assert len(calls) == 3


def test_persistent_lock_saves_full_version_and_preserves_old_last(
    tmp_path, monkeypatch
):
    previous = tmp_path / "last.pt"
    previous.write_bytes(b"previous")
    temporary = tmp_path / "last.tmp"
    temporary.write_bytes(b"complete new state")
    real_replace = Path.replace

    def locked(source, target):
        if target == previous:
            raise PermissionError("simulated persistent lock")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("latent_ar.checkpoint_io.time.sleep", lambda delay: None)
    published = publish_checkpoint(temporary, previous, 3100)
    assert published != previous
    assert published.read_bytes() == b"complete new state"
    assert previous.read_bytes() == b"previous"
    assert not temporary.exists()


def test_disk_errors_are_not_suppressed(tmp_path, monkeypatch):
    temporary = tmp_path / "last.tmp"
    temporary.write_bytes(b"complete state")

    def failed(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(Path, "replace", failed)
    with pytest.raises(OSError, match="disk failure"):
        publish_checkpoint(temporary, tmp_path / "last.pt", 1)
    assert temporary.exists()
