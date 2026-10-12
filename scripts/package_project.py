"""Build a small upload archive from an explicit source-file allowlist."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="../Tool-R1-A6000-upload.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    candidates = []
    for directory in ("src", "tests", "scripts", "configs", "docs", ".github"):
        candidates.extend(path for path in (root / directory).rglob("*") if path.is_file()
                          and "__pycache__" not in path.parts and path.suffix != ".pyc")
    candidates.extend(root / name for name in ("README.md", "LICENSE", "pyproject.toml", "CONTRACT.md", ".gitignore", ".gitattributes"))
    # Archive recorded engineering evidence only; no models, environment or data.
    evidence = root / "outputs/scripted-smoke"
    if evidence.is_dir():
        candidates.extend(evidence / name for name in ("manifest.json", "summary.json", "report.md", "traces.jsonl"))
    checksums = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(candidates):
            relative = path.relative_to(root).as_posix()
            if relative.startswith("outputs/"):
                relative = "verification/" + relative.removeprefix("outputs/")
            raw = path.read_bytes()
            if path.suffix == ".sh":
                raw = raw.replace(b"\r\n", b"\n")
            checksums[relative] = hashlib.sha256(raw).hexdigest()
            archive.writestr("Tool-R1/" + relative, raw)
        archive.writestr("Tool-R1/PACKAGE_SHA256.json", json.dumps(checksums, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for relative, expected in checksums.items():
            assert hashlib.sha256(archive.read("Tool-R1/" + relative)).hexdigest() == expected
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "files": len(checksums) + 1, "bytes": output.stat().st_size,
                      "sha256": checksum}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
