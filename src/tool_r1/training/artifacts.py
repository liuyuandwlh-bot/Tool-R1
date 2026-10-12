"""Streaming hashes for reproducible model identity without loading weights."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_merged_files(directory: str | Path, metadata: dict) -> None:
    root = Path(directory).resolve()
    files = metadata.get("merged_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Merged SFT model lacks file hashes; re-export with training.merge")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Merged model file changed or missing: {name}")
