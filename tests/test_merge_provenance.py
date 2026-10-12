import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tool_r1.training.merge import validate_merge_provenance


class MergeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        self.weights = self.adapter / "adapter_model.safetensors"
        self.weights.write_bytes(b"test adapter weights")
        self.metadata = {
            "stage": "sft", "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "base_model_revision": "a" * 40,
            "adapter_sha256": hashlib.sha256(self.weights.read_bytes()).hexdigest(),
        }

    def write_metadata(self):
        (self.adapter / "tool_r1_sft.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_remote_base_and_tokenizer_share_immutable_revision(self):
        self.write_metadata()
        metadata, load_kwargs = validate_merge_provenance(self.metadata["base_model"], self.adapter)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(load_kwargs, {"revision": "a" * 40})

    def test_mismatched_base_is_rejected(self):
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "base_model"):
            validate_merge_provenance("Qwen/Qwen2.5-1.5B-Instruct", self.adapter)

    def test_modified_adapter_is_rejected(self):
        self.write_metadata()
        self.weights.write_bytes(b"replacement adapter")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_merge_provenance(self.metadata["base_model"], self.adapter)

    def test_missing_or_mutable_remote_revision_is_rejected(self):
        for revision in (None, "main", "v1", "a" * 39):
            with self.subTest(revision=revision):
                self.metadata["base_model_revision"] = revision
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, "immutable"):
                    validate_merge_provenance(self.metadata["base_model"], self.adapter)

    def test_local_sft_base_allows_equivalent_normalized_path(self):
        base = self.root / "local-base"
        base.mkdir()
        self.metadata.update(base_model=str(base), base_model_revision=None)
        self.write_metadata()
        _, load_kwargs = validate_merge_provenance(str(base / "."), self.adapter)
        self.assertEqual(load_kwargs, {})

    def test_grpo_checkpoint_without_sft_metadata_uses_local_base(self):
        base = self.root / "merged-sft"
        base.mkdir()
        self.assertEqual(validate_merge_provenance(str(base), self.adapter), ({"stage": "grpo_export"}, {}))
        with self.assertRaisesRegex(ValueError, "local merged SFT"):
            validate_merge_provenance("Qwen/Qwen2.5-3B-Instruct", self.adapter)


if __name__ == "__main__":
    unittest.main()
