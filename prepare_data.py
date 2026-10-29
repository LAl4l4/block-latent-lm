"""固定抽样并隔离 train/dev/test，只用训练故事拟合 byte-level BPE。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from download_data import sha256

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def story_hash(text: str) -> str:
    """忽略连续空白差别，为精确文本去重生成标识。"""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def rows(paths):
    """按固定分片顺序逐批读取，每篇故事保留其官方 split 内的行号。"""
    global_index = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=["text"]
        ):
            for text in batch.column(0).to_pylist():
                yield global_index, text
                global_index += 1


def select_records(raw: Path, train_count: int, seed: int):
    """先划分验证故事，再随机选择与其不重复的训练故事。"""
    train_files = sorted(raw.glob("train-*.parquet"))
    validation_files = sorted(raw.glob("validation-*.parquet"))
    if not train_files or not validation_files:
        raise FileNotFoundError("Run download_data.py first")
    # 同一故事不能同时出现在训练、调参和最终测试中。
    validation = {}
    for index, text in rows(validation_files):
        if text and text.strip():
            digest = story_hash(text)
            validation.setdefault(
                digest,
                {"source_id": f"validation:{index}", "sha256": digest, "text": text},
            )
    validation_records = list(validation.values())
    random.Random(seed + 1).shuffle(validation_records)
    midpoint = len(validation_records) // 2
    dev, test = validation_records[:midpoint], validation_records[midpoint:]
    total_train = sum(pq.ParquetFile(path).metadata.num_rows for path in train_files)
    if not 1 <= train_count <= total_train:
        raise ValueError(f"train-count must be in [1, {total_train}]")
    # 随机抽取行号，并留出候选余量，以补足重复或无效故事被排除后的数量。
    candidate_count = min(total_train, train_count + max(1000, train_count // 10))
    order = random.Random(seed).sample(range(total_train), candidate_count)
    wanted = set(order)
    candidates = {index: text for index, text in rows(train_files) if index in wanted}
    seen = set(validation)
    train = []
    for index in order:
        text = candidates[index]
        if not text or not text.strip():
            continue
        digest = story_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        train.append({"source_id": f"train:{index}", "sha256": digest, "text": text})
        if len(train) == train_count:
            break
    if len(train) != train_count:
        raise RuntimeError(
            "Insufficient unique training candidates; reduce --train-count"
        )
    return {"train": train, "dev": dev, "test": test}, total_train


def write_split(output: Path, name: str, records, tokenizer: Tokenizer):
    """保存故事原文清单、连续 token 数组和每篇故事的起止偏移。"""
    offsets = [0]
    eos_id = tokenizer.token_to_id("<|eos|>")
    with (
        (output / f"{name}.bin").open("wb") as tokens_file,
        (output / f"{name}.jsonl").open("w", encoding="utf-8") as stories_file,
    ):
        for start in range(0, len(records), 256):
            group = records[start : start + 256]
            encodings = tokenizer.encode_batch(
                [r["text"] for r in group], add_special_tokens=False
            )
            for record, encoded in zip(group, encodings):
                # 此处处理的是完整故事，因此可以在自然末尾加入 EOS。
                # 训练切窗时只截取这份数组，不为截断窗口另外补 EOS。
                ids = np.asarray(encoded.ids + [eos_id], dtype="<u4")
                tokens_file.write(ids.tobytes())
                offsets.append(offsets[-1] + len(ids))
                stories_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.save(output / f"{name}.offsets.npy", np.asarray(offsets, dtype=np.int64))
    print(f"{name}: {len(records):,} stories, {offsets[-1]:,} tokens", flush=True)
    return {
        "stories": len(records),
        "tokens": offsets[-1],
        "selection_sha256": sha256(output / f"{name}.jsonl"),
        "tokens_sha256": sha256(output / f"{name}.bin"),
        "offsets_sha256": sha256(output / f"{name}.offsets.npy"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--train-count", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.vocab_size < 260:
        raise ValueError("Byte-level BPE needs at least 256 bytes plus special tokens")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Use a new/empty output directory to preserve reproducible prepared data"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.raw / "source.json").read_text(encoding="utf-8"))
    # 下载步骤已做 SHA-256 校验；这里先确认所有分片齐全且大小正确。
    for entry in source["files"]:
        path = args.raw / Path(entry["path"]).name
        if not path.exists() or path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Missing/incomplete raw shard: {path}")
    print(
        "Selecting stories and separating exact duplicates across splits...", flush=True
    )
    splits, total_train = select_records(args.raw, args.train_count, args.seed)
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    print("Fitting tokenizer on selected training stories only...", flush=True)
    tokenizer.train_from_iterator(
        (r["text"] for r in splits["train"]),
        trainer=trainer,
        length=len(splits["train"]),
    )
    tokenizer.save(str(args.output / "tokenizer.json"))
    metadata = {
        "format_version": 1,
        "source": source,
        "seed": args.seed,
        "official_train_rows": total_train,
        "requested_train_count": args.train_count,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_ids": {
            name: tokenizer.token_to_id(f"<|{name}|>")
            for name in ["pad", "bos", "eos", "unk"]
        },
        "tokenizer_sha256": sha256(args.output / "tokenizer.json"),
        "deduplication": "Whitespace-normalized exact SHA-256; validation excluded from train; dev/test disjoint",
        "splits": {},
    }
    for name, records in splits.items():
        metadata["splits"][name] = write_split(args.output, name, records, tokenizer)
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared data ready at {args.output}; tokenizer vocab {metadata['vocab_size']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
