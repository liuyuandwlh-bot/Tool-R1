import json
from pathlib import Path
import tempfile
import unittest

from tool_r1.training.export_checkpoint import checkpoint_adapter
from tool_r1.training.artifacts import sha256_file
from tool_r1.training.launch import require_resume_checkpoint


class ExportTests(unittest.TestCase):
    def test_resume_requires_complete_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaises(FileNotFoundError):
                require_resume_checkpoint(run)
            (run / "run.json").write_text("{}")
            (run / "checkpoints").mkdir()
            (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("20")
            checkpoint = run / "checkpoints/global_step_20"
            with self.assertRaises(FileNotFoundError):
                require_resume_checkpoint(run)
            for name in ["data.pt", "actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
                         "actor/extra_state_world_size_1_rank_0.pt", "actor/lora_adapter/adapter_model.safetensors",
                         "actor/lora_adapter/adapter_config.json"]:
                path = checkpoint / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"mock")
            self.assertEqual(require_resume_checkpoint(run), checkpoint)

    def test_exports_actual_completed_lora_checkpoint_and_exact_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "sft"
            base.mkdir()
            (base / "config.json").write_text("{}")
            metadata = {"stage": "sft", "merged": True, "dataset_sha256": "demo",
                        "merged_files": {"config.json": sha256_file(base / "config.json")}}
            (base / "tool_r1_sft.json").write_text(json.dumps(metadata))
            run = root / "run"
            adapter = run / "checkpoints/global_step_20/actor/lora_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"mock")
            (adapter / "adapter_config.json").write_text("{}")
            (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("20")
            (run / "run.json").write_text(json.dumps({"sft": metadata, "command": ["actor_rollout_ref.model.path=" + json.dumps(str(base))]}))
            found, found_base, step = checkpoint_adapter(run)
            self.assertEqual(found, adapter)
            self.assertEqual(found_base, base)
            self.assertEqual(step, 20)
            with self.assertRaises(FileNotFoundError):
                checkpoint_adapter(run, 50)
            (base / "tool_r1_sft.json").write_text("{}")
            with self.assertRaises(ValueError):
                checkpoint_adapter(run)


if __name__ == "__main__":
    unittest.main()
