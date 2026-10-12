"""Merge a PEFT adapter into an immutable HF base, including checkpoint exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from .artifacts import sha256_file


def validate_merge_provenance(base: str, adapter: str | Path) -> tuple[dict, dict]:
    """Check SFT provenance and return metadata plus pinned HF loading kwargs.

    This validation deliberately needs no Torch/Transformers imports so a
    mismatched base or altered adapter fails before allocating model memory.
    RL checkpoint adapters have no SFT manifest and use the local merged SFT
    base that training.export_checkpoint has already verified.
    """
    adapter = Path(adapter)
    metadata_path = adapter / "tool_r1_sft.json"
    local_base = Path(base).is_dir()
    if not metadata_path.exists():
        if not local_base:
            raise ValueError("An adapter without SFT metadata must use its local merged SFT base")
        return {"stage": "grpo_export"}, {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("stage") != "sft":
        raise ValueError("Invalid SFT adapter provenance stage")
    recorded_base = metadata.get("base_model")
    same_base = isinstance(recorded_base, str) and (
        Path(recorded_base).resolve() == Path(base).resolve() if local_base else recorded_base == base
    )
    if not same_base:
        raise ValueError("Merge base differs from the SFT adapter's recorded base_model")
    expected_hash = metadata.get("adapter_sha256")
    if not isinstance(expected_hash, str) or sha256_file(adapter / "adapter_model.safetensors") != expected_hash:
        raise ValueError("SFT adapter SHA256 differs from its recorded provenance")
    if local_base:
        return metadata, {}
    revision = metadata.get("base_model_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("Remote SFT base requires a recorded immutable 40-character commit revision")
    return metadata, {"revision": revision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="For RL adapters, use the merged SFT model")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    adapter, output = Path(args.adapter).resolve(), Path(args.output).resolve()
    if not (adapter / "adapter_config.json").exists() or not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Not an exported PEFT adapter: {adapter}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    metadata, revision_kwargs = validate_merge_provenance(args.base, adapter)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                device_map="cpu", trust_remote_code=False, **revision_kwargs)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False, **revision_kwargs)
    tokenizer.save_pretrained(output)
    metadata.update(merged=True, merge_base=args.base, adapter=str(adapter),
                    adapter_sha256=sha256_file(adapter / "adapter_model.safetensors"),
                    merged_files={path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()})
    name = "tool_r1_sft.json" if metadata.get("stage") == "sft" else "tool_r1_export.json"
    (output / name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Merged model: {output}")


if __name__ == "__main__":
    main()
