"""从本机 Git LFS 历史快照恢复单个归档文件，不触碰 Git 索引或活动文件。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_path", help="paper/artifacts.json 中的 original_path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "paper/artifacts.json").read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in manifest["artifacts"]
        if entry["original_path"] == args.original_path.replace("\\", "/")
    ]
    if len(matches) != 1:
        parser.error("Artifact path is missing or ambiguous; see paper/artifacts.json")
    entry = matches[0]
    oid = entry["sha256"]
    source = root / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    if not source.is_file():
        parser.error(
            "Local LFS object is unavailable; do not delete .git/lfs without an external backup"
        )
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != oid or source.stat().st_size != entry["bytes"]:
        raise ValueError("Archived artifact failed integrity verification")
    output = args.output.resolve()
    if output.exists():
        parser.error(
            "Output already exists; choose a new path to avoid overwriting a checkpoint"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, output.open("xb") as dst:
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
    print(f"Restored: {output}")


if __name__ == "__main__":
    main()
