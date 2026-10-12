"""Read-only, standard-library environment checks for the single-A6000 recipe.

Package metadata is inspected without importing the ML stack in this process.
An optional short-lived child checks PyTorch CUDA/BF16 on the selected GPU.
No credentials are inspected and subprocess stderr is never copied to reports.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3
VERL_COMMIT = "f9c855f7cf04d603c9546bc01776c74806a879c1"
EXPECTED_PACKAGES = {"torch": "2.8.0", "vllm": "0.11.0", "transformers": "4.57.1", "peft": "0.17.1"}
MIN_DRIVER = (570, 26)


def _python_info() -> dict:
    return {"version": platform.python_version(), "major": sys.version_info.major,
            "minor": sys.version_info.minor, "executable": sys.executable}


def _ram_info() -> dict:
    physical, available, limits = None, None, []
    if platform.system() == "Linux":
        try:
            fields = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                fields[name] = int(value.strip().split()[0]) * 1024
            physical, available = fields.get("MemTotal"), fields.get("MemAvailable")
        except (OSError, ValueError, IndexError):
            pass
        for location in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                raw = Path(location).read_text().strip()
                if raw != "max" and 0 < int(raw) < (physical or 2 ** 60):
                    limits.append(int(raw))
            except (OSError, ValueError):
                pass
    elif platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                physical, available = status.total_phys, status.avail_phys
        except (AttributeError, OSError):
            pass
    effective = min([physical, *limits]) if physical is not None else (min(limits) if limits else None)
    return {"physical_bytes": physical, "available_bytes": available,
            "cgroup_limit_bytes": min(limits) if limits else None,
            "effective_total_bytes": effective,
            "effective_total_gib": round(effective / GIB, 2) if effective is not None else None}


def _storage_info(workspace: str | Path) -> dict:
    requested = Path(workspace).expanduser().resolve()
    existing = requested
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    if existing.is_file():
        existing = existing.parent
    try:
        usage = shutil.disk_usage(existing)
        return {"workspace": str(requested), "checked_path": str(existing),
                "total_bytes": usage.total, "free_bytes": usage.free,
                "free_gib": round(usage.free / GIB, 2)}
    except OSError as error:
        return {"workspace": str(requested), "checked_path": str(existing),
                "total_bytes": None, "free_bytes": None, "free_gib": None,
                "error": type(error).__name__}


def _query_gpus() -> tuple[list[dict], str | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], "nvidia-smi was not found"
    command = [executable, "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], f"nvidia-smi query failed ({type(error).__name__})"
    if result.returncode:
        return [], f"nvidia-smi query returned exit code {result.returncode}"
    devices = []
    try:
        for row in csv.reader(io.StringIO(result.stdout)):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError("Unexpected NVIDIA CSV schema")
            index, uuid, name, memory, driver = (part.strip() for part in row)
            devices.append({"index": int(index), "uuid": uuid, "name": name,
                            "memory_total_mib": int(memory), "driver_version": driver})
    except ValueError:
        return [], "nvidia-smi returned an unrecognized GPU table"
    return devices, None


def _select_gpu(devices: list[dict], visibility: str | None) -> tuple[dict | None, str, str | None]:
    token = "0" if visibility is None else visibility.split(",", 1)[0].strip()
    if token in {"", "-1"}:
        return None, token, "CUDA_VISIBLE_DEVICES disables the first CUDA device"
    if token.isdecimal():
        matches = [device for device in devices if device["index"] == int(token)]
    elif token.startswith("GPU-"):
        # CUDA supports unambiguous UUID prefixes as well as complete UUIDs.
        matches = [device for device in devices if device["uuid"].startswith(token)]
    else:
        return None, token, "First CUDA_VISIBLE_DEVICES entry must be a physical GPU index or GPU UUID"
    if len(matches) != 1:
        return None, token, "Selected CUDA GPU was not found or its UUID prefix is ambiguous"
    return dict(matches[0]), token, None


def _package_info() -> dict:
    packages = {}
    for name, expected in {**EXPECTED_PACKAGES, "verl": "0.7.0 (pinned Git commit)"}.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if name == "verl":
            entry = {"version": version, "expected": expected, "expected_commit": VERL_COMMIT, "commit": None}
            try:
                raw = importlib.metadata.distribution(name).read_text("direct_url.json")
                if raw:
                    entry["commit"] = json.loads(raw).get("vcs_info", {}).get("commit_id")
            except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
                pass
            entry["matches"] = version in {"0.7.0", "0.7.0.dev", "0.7.0.dev0"} and entry["commit"] == VERL_COMMIT
        else:
            normalized = version.split("+", 1)[0] if name == "torch" and version is not None else version
            entry = {"version": version, "expected": expected, "matches": normalized == expected}
        packages[name] = entry
    return packages


_TORCH_PROBE = '''import json, os, sys
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
result = {}
try:
    import torch
    result["torch_version"] = str(torch.__version__)
    result["cuda_version"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["device_count"] = int(torch.cuda.device_count())
    result["bf16_supported"] = False
    if result["cuda_available"]:
        torch.cuda.set_device(0)
        result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        result["device_name"] = torch.cuda.get_device_name(0)
        result["device_memory_mib"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
except Exception as error:
    result["error"] = type(error).__name__
print("TOOL_R1_PREFLIGHT=" + json.dumps(result, sort_keys=True))
'''


def _probe_torch(gpu_uuid: str | None) -> dict:
    command = [sys.executable, "-c", _TORCH_PROBE]
    if gpu_uuid is not None:
        command.append(gpu_uuid)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"checked": True, "error": type(error).__name__}
    if result.returncode:
        return {"checked": True, "error": f"Probe process exited with code {result.returncode}"}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("TOOL_R1_PREFLIGHT="):
            try:
                report = json.loads(line.split("=", 1)[1])
                if isinstance(report, dict):
                    return dict(report, checked=True)
            except ValueError:
                break
    return {"checked": True, "error": "Probe process did not return a JSON report"}


def _driver_version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)+", value):
        return ()
    parts = tuple(int(part) for part in value.split("."))
    return parts + (0,) * max(0, 2 - len(parts))


def inspect_environment(require_a6000: bool = False, require_stack: bool = False, workspace: str | Path = ".") -> dict:
    """Return a JSON-serializable read-only environment report.

    Ordinary mode supports local CPU-only development. Strict A6000 mode checks
    the server recipe's hardware/OS thresholds. Stack mode additionally checks
    installed versions, the verl source commit, and CUDA/BF16 in a child process.
    Only the first CUDA_VISIBLE_DEVICES entry selects the single training GPU.
    """
    errors, warnings = [], []
    host = {"system": platform.system(), "machine": platform.machine(), "release": platform.release()}
    python = _python_info()
    cpu = {"logical_count": os.cpu_count(), "available_logical_count": None}
    try:
        cpu["available_logical_count"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu["available_logical_count"] = cpu["logical_count"]
    ram, storage = _ram_info(), _storage_info(workspace)
    devices, gpu_error = _query_gpus()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected, token, selection_error = _select_gpu(devices, visibility)
    packages = _package_info()
    strict = errors if require_a6000 else warnings

    if require_a6000:
        if host["system"] != "Linux":
            errors.append("A6000 training requires Linux; ordinary local checks remain available on other systems")
        if host["machine"].lower() not in {"x86_64", "amd64"}:
            errors.append("The pinned A6000 server stack requires x86_64")
        if (python["major"], python["minor"]) != (3, 11):
            errors.append("The pinned A6000 server stack requires Python 3.11")
    elif (python["major"], python["minor"]) < (3, 10):
        errors.append("The runtime core requires Python 3.10 or newer")
    if gpu_error:
        strict.append(gpu_error)
    if selected is None:
        if selection_error and (not gpu_error or visibility is not None):
            strict.append(selection_error)
    else:
        if not re.search(r"\bA6000\b", selected["name"], flags=re.IGNORECASE):
            strict.append("Selected GPU is not an NVIDIA RTX A6000")
        if selected["memory_total_mib"] < 45000:
            strict.append("Selected GPU has less than 45000 MiB total memory")
        if _driver_version(selected["driver_version"]) < MIN_DRIVER:
            strict.append("NVIDIA driver 570.26 or newer is required for the CUDA 12.8 server recipe")
    if visibility is not None and len(visibility.split(",")) > 1:
        warnings.append("Multiple CUDA_VISIBLE_DEVICES entries are present; this recipe checks and uses the first selected GPU only")

    total_ram = ram.get("effective_total_bytes")
    if total_ram is None:
        strict.append("Could not determine system RAM")
    elif total_ram < 32 * GIB:
        strict.append("At least 32 GiB effective system RAM is required for the A6000 recipe")
    elif total_ram < 64 * GIB:
        warnings.append("64 GiB or more system RAM is recommended; CPU offload may need additional headroom")
    free = storage.get("free_bytes")
    if free is None:
        strict.append("Could not determine free disk space in the workspace filesystem")
    elif free < 60 * GIB:
        strict.append("At least 60 GiB free workspace disk space is required")
    elif free < 150 * GIB:
        warnings.append("150 GiB free workspace disk space is recommended for models, checkpoints, and logs")
    available_cpu = cpu["available_logical_count"]
    if available_cpu is not None and available_cpu < 8:
        warnings.append("At least 8 available CPU threads are recommended")

    cuda = {"checked": False}
    if require_stack:
        for name, package in packages.items():
            if not package["matches"]:
                errors.append(f"{name}: install the pinned server requirement ({package['expected']})" +
                              (" from the required Git commit" if name == "verl" else ""))
        if packages["torch"]["version"] is not None:
            cuda = _probe_torch(selected["uuid"] if selected else None)
            if cuda.get("error"):
                errors.append("PyTorch CUDA probe failed: " + str(cuda["error"]))
            if cuda.get("cuda_available") is not True:
                errors.append("PyTorch CUDA is not available on the selected GPU")
            if cuda.get("bf16_supported") is not True:
                errors.append("The selected GPU/PyTorch stack does not report BF16 support")
            if selected and cuda.get("device_count") != 1:
                errors.append("The isolated PyTorch probe must see exactly the selected GPU")
            if cuda.get("cuda_version") not in {None, "12.8"}:
                warnings.append("PyTorch uses a CUDA runtime other than the recipe's CUDA 12.8; verify wheel/driver compatibility")
        else:
            cuda = {"checked": False, "reason": "PyTorch is not installed"}
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "mode": {"require_a6000": require_a6000, "require_stack": require_stack},
            "platform": host, "python": python, "cpu": cpu, "ram": ram, "storage": storage,
            "gpus": devices, "selected_gpu": selected, "cuda_visible_devices": visibility,
            "selected_cuda_token": token, "packages": packages, "cuda": cuda,
            "thresholds": {"gpu_memory_mib_min": 45000, "driver_min": "570.26",
                           "ram_gib_min": 32, "ram_gib_recommended": 64,
                           "disk_free_gib_min": 60, "disk_free_gib_recommended": 150},
            "note": "Passing preflight checks prerequisites only; run the 20-update capacity smoke test to measure actual VRAM and throughput."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-a6000", action="store_true")
    parser.add_argument("--require-stack", action="store_true")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", help="Optional JSON report file")
    args = parser.parse_args()
    report = inspect_environment(args.require_a6000, args.require_stack, args.workspace)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
