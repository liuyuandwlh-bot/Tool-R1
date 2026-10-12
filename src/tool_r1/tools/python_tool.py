"""Bounded calculation tools: portable restricted syntax or isolated Docker.

``restricted`` is a deterministic Python-syntax subset, not arbitrary Python.
``docker`` runs real Python in a fresh, resource-limited, unprivileged container.
Docker must already be installed and the pinned image pre-pulled by the user.
No network, GPU, host mounts, host environment, or Docker socket is exposed to
model code. The host's Docker daemon remains part of the trusted boundary.
"""

from __future__ import annotations

import math
import secrets
import subprocess
import threading
import time

from ._restricted import Interpreter, MAX_SOURCE_BYTES, RestrictedError

DOCKER_IMAGE = "python:3.11.11-slim-bookworm"
MAX_OUTPUT_CHARS = 65_536


def run_python(code, backend="restricted", timeout=5.0, max_output_chars=4096):
    """Return ``{ok, result, error, backend}`` without executing code on host.

    Timeouts include interpreter execution or Docker startup and execution.
    The restricted backend checks time cooperatively between bounded operations.
    Production callers should select Docker for full Python semantics.
    """
    def response(ok, result="", error=None):
        return {"ok": ok, "result": result, "error": error, "backend": backend}

    if backend not in ("restricted", "docker"):
        return response(False, error="unknown Python backend")
    if not isinstance(code, str):
        return response(False, error="code must be a string")
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not 0 < timeout <= 60 or not math.isfinite(timeout):
        return response(False, error="timeout must be greater than zero and at most 60 seconds")
    if type(max_output_chars) is not int or not 1 <= max_output_chars <= MAX_OUTPUT_CHARS:
        return response(False, error="invalid output size limit")
    try:
        if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
            return response(False, error="source size limit exceeded")
        if backend == "restricted":
            return response(True, Interpreter(timeout, max_output_chars).run(code))
        result, error = _run_docker(code, timeout, max_output_chars)
        return response(error is None, result, error)
    except (RestrictedError, ArithmeticError, TypeError, LookupError, ValueError, RecursionError) as exc:
        return response(False, error=f"{type(exc).__name__}: {str(exc)[:300]}")


def _docker_command(name):
    return [
        "docker", "run", "--pull=never", "--rm", "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--memory", "256m",
        "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "32",
        "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "-i", DOCKER_IMAGE, "python", "-I", "-S", "-u", "-",
    ]


def _remove_container(name):
    # Name is freshly generated internally; never accept a model-supplied name.
    try:
        subprocess.run(["docker", "rm", "-f", name], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _stop_process(process):
    try:
        process.terminate()
        process.wait(timeout=1)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
            process.wait(timeout=1)
        except (OSError, subprocess.SubprocessError):
            pass


def _run_docker(code, timeout, max_output_chars):
    name = "tool-r1-" + secrets.token_hex(12)
    try:
        process = subprocess.Popen(_docker_command(name), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False)
    except OSError:
        return "", "Docker is unavailable; install Docker and pre-pull " + DOCKER_IMAGE

    data = bytearray()
    byte_limit = 4 * max_output_chars
    output_exceeded = threading.Event()
    reader_done = threading.Event()
    writer_error = threading.Event()
    deadline = time.monotonic() + timeout

    def read_output():
        try:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                remaining = byte_limit - len(data)
                data.extend(chunk[:max(0, remaining)])
                if len(chunk) > remaining or len(data.decode("utf-8", errors="replace")) > max_output_chars:
                    output_exceeded.set()
                    break
        except (OSError, ValueError):
            pass
        finally:
            reader_done.set()

    def write_input():
        try:
            process.stdin.write(code.encode("utf-8"))
            process.stdin.close()
        except (OSError, ValueError):
            writer_error.set()

    reader = threading.Thread(target=read_output, daemon=True)
    writer = threading.Thread(target=write_input, daemon=True)
    reader.start()
    writer.start()
    error = None
    try:
        while True:
            if output_exceeded.is_set():
                error = "output size limit exceeded"
                break
            if time.monotonic() > deadline:
                error = "execution timeout exceeded"
                break
            if process.poll() is not None and reader_done.is_set():
                break
            time.sleep(0.01)
    finally:
        if error is not None or process.poll() is None:
            _stop_process(process)
            _remove_container(name)
        reader.join(timeout=1)
        writer.join(timeout=1)
        # --rm handles normal termination; explicit cleanup covers early failures.
        if process.poll() not in (None, 0) and error is None:
            _remove_container(name)
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
    text = data.decode("utf-8", errors="replace")[:max_output_chars]
    if error is None and output_exceeded.is_set():
        error = "output size limit exceeded"
    if error is None and process.returncode != 0:
        error = "Python container exited unsuccessfully; see result for details"
    if error is None and writer_error.is_set():
        error = "failed to deliver source to Python container"
    return text, error
