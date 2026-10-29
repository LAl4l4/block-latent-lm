"""下载官方 TinyStories Parquet 数据，支持固定版本、断点续传和校验。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = "roneneldan/TinyStories"
BASE = "https://huggingface.co"


def request(url: str, **headers):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "latent-ar-research/0.1", **headers}
        ),
        timeout=60,
    )


def sha256(path: Path) -> str:
    """分块计算校验和，不一次性把大文件读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, size: int, checksum: str | None):
    """已完成的文件先校验；未完成的文件从 .partial 的末尾继续下载。"""
    if path.exists() and path.stat().st_size == size:
        if checksum is None or sha256(path) == checksum:
            print(f"Verified existing {path.name}", flush=True)
            return
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                partial.write_bytes(b"")
                offset = 0
            if offset < size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                # 使用新查询避免重用已经过期的 CDN 签名跳转链接。
                with request(
                    url + f"?download=true&ts={time.time_ns()}", **headers
                ) as response:
                    if offset and response.status != 206:
                        offset = 0
                    if response.status == 206:
                        expected = f"bytes {offset}-"
                        if not response.headers.get("Content-Range", "").startswith(
                            expected
                        ):
                            raise RuntimeError(
                                "Server returned an unexpected byte range"
                            )
                    print(
                        f"Downloading {path.name}: {offset / 1e6:.1f}/{size / 1e6:.1f} MB",
                        flush=True,
                    )
                    last_report = time.monotonic()
                    with partial.open("ab" if offset else "wb") as output:
                        while block := response.read(4 << 20):
                            output.write(block)
                            offset += len(block)
                            if time.monotonic() - last_report >= 10:
                                print(
                                    f"  {path.name}: {offset / 1e6:.1f}/{size / 1e6:.1f} MB",
                                    flush=True,
                                )
                                last_report = time.monotonic()
            if partial.stat().st_size != size:
                raise RuntimeError(
                    f"Incomplete download: {partial.stat().st_size} != {size}"
                )
            if checksum and sha256(partial) != checksum:
                partial.unlink()
                raise RuntimeError("SHA-256 mismatch; discarded corrupt partial file")
            partial.replace(path)
            print(f"Verified {path.name}", flush=True)
            return
        except Exception as error:
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}/4: {error}", flush=True)
            time.sleep(min(2**attempt, 8))


def main():
    """先记录官方 commit 和分片校验信息，再逐个下载原始文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--revision",
        default=None,
        help="HF commit; existing snapshot is reused by default",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_path = args.output / "source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if args.revision and args.revision != source["revision"]:
            raise ValueError("Use a new output directory for a different revision")
    else:
        with request(
            f"{BASE}/api/datasets/{REPO}/revision/{args.revision or 'main'}"
        ) as stream:
            metadata = json.load(stream)
        revision = metadata["sha"]
        with request(f"{BASE}/api/datasets/{REPO}/tree/{revision}/data") as stream:
            entries = json.load(stream)
        files = [
            {
                "path": entry["path"],
                "size": entry["size"],
                "sha256": entry.get("lfs", {}).get("oid"),
            }
            for entry in entries
            if entry["path"].endswith(".parquet")
        ]
        if not files or not any("validation-" in f["path"] for f in files):
            raise RuntimeError("No complete train/validation parquet listing found")
        source = {
            "repo": REPO,
            "revision": revision,
            "license": "cdla-sharing-1.0",
            "url": f"{BASE}/datasets/{REPO}",
            "files": sorted(files, key=lambda f: f["path"]),
        }
        source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(
        f"Snapshot {source['revision']}; total {sum(f['size'] for f in source['files']) / 1e9:.3f} GB",
        flush=True,
    )
    for entry in source["files"]:
        download(
            f"{BASE}/datasets/{REPO}/resolve/{source['revision']}/{entry['path']}",
            args.output / Path(entry["path"]).name,
            entry["size"],
            entry["sha256"],
        )
    with request(
        f"{BASE}/datasets/{REPO}/raw/{source['revision']}/README.md"
    ) as stream:
        (args.output / "DATASET_CARD.md").write_bytes(stream.read())
    print(
        "All official train and validation shards downloaded and verified.", flush=True
    )


if __name__ == "__main__":
    main()
