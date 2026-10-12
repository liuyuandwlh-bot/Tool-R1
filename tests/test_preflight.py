import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tool_r1 import preflight


GPU = {"index": 0, "uuid": "GPU-aaaa-0000", "name": "NVIDIA RTX A6000", "memory_total_mib": 49140, "driver_version": "570.26"}


def packages():
    result = {name: {"version": version, "expected": version, "matches": True} for name, version in preflight.EXPECTED_PACKAGES.items()}
    result["torch"]["version"] = "2.8.0+cu128"
    result["verl"] = {"version": "0.7.0.dev0", "expected": "0.7.0 (pinned Git commit)",
                      "commit": preflight.VERL_COMMIT, "expected_commit": preflight.VERL_COMMIT, "matches": True}
    return result


class PreflightTests(unittest.TestCase):
    def inspect(self, *, system="Linux", machine="x86_64", python_minor=11, gpus=None, visibility=None,
                ram=64, disk=150, strict=True, stack=False, package_info=None, cuda=None):
        with ExitStack() as context:
            context.enter_context(patch.dict(os.environ, {}, clear=True))
            if visibility is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = visibility
            for name, value in (
                ("platform.system", system), ("platform.machine", machine), ("platform.release", "mock"),
                ("_python_info", {"version": f"3.{python_minor}.0", "major": 3, "minor": python_minor, "executable": "mock-python"}),
                ("_ram_info", {"effective_total_bytes": ram * preflight.GIB}),
                ("_storage_info", {"free_bytes": disk * preflight.GIB}),
                ("_query_gpus", (gpus if gpus is not None else [dict(GPU)], None)),
                ("_package_info", package_info or packages()),
                ("_probe_torch", cuda or {"checked": True, "cuda_available": True, "bf16_supported": True, "device_count": 1, "cuda_version": "12.8"}),
            ):
                context.enter_context(patch("tool_r1.preflight." + name, return_value=value))
            context.enter_context(patch("tool_r1.preflight.os.cpu_count", return_value=16))
            context.enter_context(patch("tool_r1.preflight.os.sched_getaffinity", return_value=set(range(16)), create=True))
            return preflight.inspect_environment(strict, stack, "mock-workspace")

    def test_supported_server_and_inclusive_thresholds(self):
        self.assertTrue(self.inspect(stack=True)["ok"])
        minimal = self.inspect(ram=32, disk=60, gpus=[dict(GPU, memory_total_mib=45000)])
        self.assertTrue(minimal["ok"])
        self.assertEqual(len(minimal["warnings"]), 2)

    def test_strict_hardware_platform_and_resources(self):
        result = self.inspect(system="Windows", machine="arm64", python_minor=12, ram=31, disk=59,
                              gpus=[dict(GPU, name="RTX 4090", memory_total_mib=24000, driver_version="550.90")])
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["errors"]), 8)

    def test_ordinary_windows_check_is_not_blocked_by_gpu_or_stack(self):
        result = self.inspect(system="Windows", python_minor=12, ram=16, disk=20, gpus=[], strict=False)
        self.assertTrue(result["ok"])
        self.assertTrue(result["warnings"])
        self.assertFalse(result["cuda"]["checked"])

    def test_selected_numeric_first_entry_and_uuid_prefix(self):
        wrong = dict(GPU, name="RTX 4090", memory_total_mib=24000)
        wanted = dict(GPU, index=1, uuid="GPU-bbbb-1111")
        result = self.inspect(gpus=[wrong, wanted], visibility="1,0")
        self.assertTrue(result["ok"])
        self.assertEqual(result["selected_gpu"]["index"], 1)
        result = self.inspect(gpus=[wrong, wanted], visibility="GPU-bbbb")
        self.assertTrue(result["ok"])
        self.assertEqual(result["selected_gpu"]["uuid"], "GPU-bbbb-1111")

    def test_disabled_unknown_and_ambiguous_gpu_selectors(self):
        for visibility in ("", "-1", "9", "invalid", "GPU-not-found"):
            self.assertFalse(self.inspect(visibility=visibility)["ok"], visibility)
        self.assertFalse(self.inspect(gpus=[GPU, dict(GPU, index=1, uuid="GPU-aaaa-1111")], visibility="GPU-aaaa")["ok"])

    def test_stack_failures_cuda_bf16_and_commit_are_blocking(self):
        wrong = packages()
        wrong["verl"]["matches"] = False
        wrong["verl"]["commit"] = "wrong"
        report = self.inspect(stack=True, package_info=wrong,
                              cuda={"checked": True, "cuda_available": False, "bf16_supported": False, "device_count": 0})
        self.assertFalse(report["ok"])
        self.assertTrue(any("Git commit" in error for error in report["errors"]))
        self.assertTrue(any("CUDA is not available" in error for error in report["errors"]))
        self.assertTrue(any("BF16" in error for error in report["errors"]))

    def test_metadata_accepts_torch_local_suffix_and_requires_verl_commit(self):
        versions = {**preflight.EXPECTED_PACKAGES, "torch": "2.8.0+cu128", "verl": "0.7.0.dev0"}
        distribution = SimpleNamespace(read_text=lambda _: json.dumps({"vcs_info": {"commit_id": preflight.VERL_COMMIT}}))
        with patch("tool_r1.preflight.importlib.metadata.version", side_effect=lambda name: versions[name]), \
                patch("tool_r1.preflight.importlib.metadata.distribution", return_value=distribution):
            report = preflight._package_info()
        self.assertTrue(all(entry["matches"] for entry in report.values()))
        distribution.read_text = lambda _: json.dumps({"vcs_info": {"commit_id": "bad"}})
        with patch("tool_r1.preflight.importlib.metadata.version", side_effect=lambda name: versions[name]), \
                patch("tool_r1.preflight.importlib.metadata.distribution", return_value=distribution):
            self.assertFalse(preflight._package_info()["verl"]["matches"])

    def test_query_gpu_csv_and_failure_do_not_expose_stderr(self):
        with patch("tool_r1.preflight.shutil.which", return_value="nvidia-smi"), \
                patch("tool_r1.preflight.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="1, GPU-test, NVIDIA RTX A6000, 49140, 570.26\n")):
            devices, error = preflight._query_gpus()
        self.assertIsNone(error)
        self.assertEqual(devices[0]["index"], 1)
        with patch("tool_r1.preflight.shutil.which", return_value="nvidia-smi"), \
                patch("tool_r1.preflight.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="SECRET_SENTINEL")):
            self.assertNotIn("SECRET_SENTINEL", str(preflight._query_gpus()))

    def test_cuda_probe_runs_in_child_and_uses_selected_uuid(self):
        payload = {"cuda_available": True, "bf16_supported": True, "device_count": 1}
        completed = SimpleNamespace(returncode=0, stdout="noise\nTOOL_R1_PREFLIGHT=" + json.dumps(payload), stderr="not included")
        with patch("tool_r1.preflight.subprocess.run", return_value=completed) as run:
            result = preflight._probe_torch("GPU-selected")
        self.assertTrue(result["checked"])
        self.assertEqual(run.call_args.args[0][-1], "GPU-selected")
        self.assertEqual(run.call_args.args[0][1], "-c")
        with patch("tool_r1.preflight.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 45)):
            self.assertEqual(preflight._probe_torch(None)["error"], "TimeoutExpired")

    def test_cli_json_output_and_exit_status(self):
        report = {"ok": False, "errors": ["mock"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            with patch("sys.argv", ["preflight", "--require-a6000", "--require-stack", "--output", str(path)]), \
                    patch("tool_r1.preflight.inspect_environment", return_value=report) as inspect, redirect_stdout(io.StringIO()):
                self.assertEqual(preflight.main(), 1)
            self.assertEqual(json.loads(path.read_text()), report)
            inspect.assert_called_once_with(True, True, ".")


if __name__ == "__main__":
    unittest.main()
