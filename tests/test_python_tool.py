"""Meaningful isolation, bounds and arithmetic checks, requiring no Docker."""

import io
import os
import unittest
from unittest import mock

from tool_r1.tools.python_tool import DOCKER_IMAGE, run_python


class RestrictedPythonTests(unittest.TestCase):
    def result(self, code):
        response = run_python(code)
        self.assertTrue(response["ok"], response)
        self.assertIsNone(response["error"])
        return response["result"]

    def test_numerical_correctness_and_dataset_demos(self):
        code = "quantities=[2,3,5]\nprices=[10,20,4]\nprint(sum(q*p for q,p in zip(quantities,prices)))"
        self.assertEqual(self.result(code), "100\n")
        self.assertEqual(self.result("print(round(100*(132-120)/120, 4))"), "10.0\n")

    def test_lists_dicts_tuples_conditions_and_loops(self):
        code = "a={'x': [3,1,2]}\ntotal=0\nfor i,v in enumerate(sorted(a['x'])):\n if i >= 1:\n  total += v\nprint((total,a['x'][1:]))"
        self.assertEqual(self.result(code), "(5, [1, 2])\n")
        self.assertEqual(self.result("print([i*i for i in range(6) if i%2 == 0])"), "[0, 4, 16]\n")

    def test_comprehension_scope_is_local(self):
        self.assertEqual(self.result("x=9\na=[x for x in range(3)]\nprint(x,a)"), "9 [0, 1, 2]\n")

    def test_no_host_file_network_or_object_access(self):
        attacks = [
            "import os\nprint(os.listdir('.'))", "open('/etc/passwd').read()",
            "__import__('socket').socket()", "print((1).__class__)",
            "eval('1+1')", "exec('print(1)')", "print(globals())",
            "print(getattr(1,'__class__'))", "f=lambda x:x\nprint(f(2))",
            "def f():\n return 1\nprint(f())", "class A:\n pass",
            "a=[]\na.append(a)", "a=[1]\na[0]=2", "print(**{'end':'x'})",
            "if False:\n import os", "if False:\n print(open('/etc/passwd'))",
        ]
        for code in attacks:
            with self.subTest(code=code):
                with mock.patch("builtins.eval", side_effect=AssertionError("host eval called")), mock.patch("builtins.exec", side_effect=AssertionError("host exec called")), mock.patch("builtins.open", side_effect=AssertionError("file access attempted")):
                    response = run_python(code)
                self.assertFalse(response["ok"], response)

    def test_bounds_prevent_large_allocations_and_unbounded_execution(self):
        attacks = [
            "while True:\n pass", "print(2**1000000000)",
            "print(10**1024**1024)", "print(range(10**100))",
            "print([0]*1000000000)", "print('x'*1000000000)",
            "print([i for i in range(4096) for j in range(4096)])",
            "a='x'*16000\nprint([a]*4000)",
            "print(zip(range(4000), range(4000)))",
            "for i in range(4000):\n print('x'*100)",
            "print(1e999)", "print((-2)**0.5)", "print(1 << 100000000)",
        ]
        for code in attacks:
            with self.subTest(code=code):
                self.assertFalse(run_python(code)["ok"])

    def test_ast_source_and_depth_limits(self):
        for code in ("a=1\n" * 5000, "print(" + "1+" * 1000 + "1)"):
            self.assertFalse(run_python(code)["ok"])

    def test_step_and_clock_limits(self):
        response = run_python("a=0\nfor i in range(4096):\n for j in range(4096):\n  a+=1")
        self.assertFalse(response["ok"])
        self.assertIn("step limit", response["error"])
        with mock.patch("tool_r1.tools._restricted.time.monotonic", side_effect=[0, 10]):
            response = run_python("print(1)", timeout=1)
        self.assertFalse(response["ok"])
        self.assertIn("timeout", response["error"])

    def test_output_limits_and_invalid_options(self):
        self.assertEqual(run_python("print(1234)", max_output_chars=5)["result"], "1234\n")
        self.assertFalse(run_python("print(12345)", max_output_chars=5)["ok"])
        for kwargs in ({"timeout": 0}, {"timeout": float("nan")}, {"timeout": 10**1000}, {"timeout": True}, {"max_output_chars": 0}, {"backend": "host"}):
            self.assertFalse(run_python("print(1)", **kwargs)["ok"])
        self.assertFalse(run_python(None)["ok"])


class FakeProcess:
    def __init__(self, output=b"", running=False):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(output)
        self.returncode = None if running else 0
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class DockerPythonTests(unittest.TestCase):
    def test_isolation_command_and_no_shell_source_interpolation(self):
        process = FakeProcess(output=b"6\n")
        with mock.patch("tool_r1.tools.python_tool.subprocess.Popen", return_value=process) as popen:
            response = run_python("print(sum([1,2,3]))", backend="docker")
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["result"], "6\n")
        command = popen.call_args.args[0]
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--user") + 1], "65534:65534")
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("--pull=never", command)
        self.assertIn(DOCKER_IMAGE, command)
        self.assertNotIn("print(sum([1,2,3]))", command)
        self.assertNotIn("-v", command)
        self.assertNotIn("--env", command)

    def test_timeout_forces_cleanup_of_only_owned_container(self):
        process = FakeProcess(running=True)
        with mock.patch("tool_r1.tools.python_tool.subprocess.Popen", return_value=process), mock.patch("tool_r1.tools.python_tool.subprocess.run") as cleanup:
            response = run_python("while True: pass", backend="docker", timeout=0.01)
        self.assertFalse(response["ok"])
        self.assertIn("timeout", response["error"])
        self.assertTrue(process.terminated)
        command = cleanup.call_args.args[0]
        self.assertEqual(command[:3], ["docker", "rm", "-f"])
        self.assertRegex(command[3], r"^tool-r1-[0-9a-f]{24}$")

    def test_output_limit_terminates_and_cleans_up(self):
        process = FakeProcess(output=b"x" * 100000, running=True)
        with mock.patch("tool_r1.tools.python_tool.subprocess.Popen", return_value=process), mock.patch("tool_r1.tools.python_tool.subprocess.run") as cleanup:
            response = run_python("print('x'*100000)", backend="docker", max_output_chars=100)
        self.assertFalse(response["ok"])
        self.assertLessEqual(len(response["result"]), 100)
        self.assertIn("output size", response["error"])
        self.assertTrue(process.terminated)
        cleanup.assert_called_once()

    def test_missing_docker_does_not_execute_on_host(self):
        with mock.patch("tool_r1.tools.python_tool.subprocess.Popen", side_effect=FileNotFoundError):
            response = run_python("print(1)", backend="docker")
        self.assertFalse(response["ok"])
        self.assertIn("Docker is unavailable", response["error"])

    @unittest.skipUnless(os.environ.get("TOOL_R1_TEST_DOCKER") == "1", "set TOOL_R1_TEST_DOCKER=1 after pre-pulling the image")
    def test_real_docker_execution_and_network_isolation(self):
        response = run_python("print(sum([1,2,3]))", backend="docker", timeout=20)
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["result"], "6\n")
        code = "import socket\ns=socket.socket()\ns.settimeout(1)\ns.connect(('1.1.1.1',80))"
        self.assertFalse(run_python(code, backend="docker", timeout=20)["ok"])


if __name__ == "__main__":
    unittest.main()
