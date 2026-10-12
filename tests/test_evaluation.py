import unittest

from tool_r1.evaluation import paired_bootstrap, summarize


def trace(sample_id, correct, calls, budget=4, task="compose"):
    return {"sample_id": sample_id, "task_type": task, "budget": budget,
            "correctness": correct, "reward": float(correct), "cost": min(calls / 6, 1),
            "metrics": {"requests": calls, "admitted_calls": calls, "executed_calls": calls,
                        "invalid_requests": 0, "execution_failures": 0, "budget_violations": 0},
            "evaluation_kind": "scripted_smoke", "terminal_reason": "final"}


class EvaluationTests(unittest.TestCase):
    def test_grouped_metrics_and_scripted_provenance(self):
        rows = [trace("a", True, 2), trace("b", False, 3), trace("c", True, 0, 0, "direct")]
        report = summarize(rows)
        self.assertAlmostEqual(report["overall"]["accuracy"], 2 / 3)
        self.assertEqual(report["per_budget"]["4"]["accuracy"], 0.5)
        self.assertEqual(report["per_task"]["direct"]["mean_tool_calls"], 0)
        self.assertEqual(report["overall"]["mean_calls_on_correct"], 1)
        self.assertEqual(report["evaluation_kinds"], {"scripted_smoke": 3})
        self.assertIsNone(report["overall"]["mean_generated_tokens"])
        self.assertIsNone(report["per_task"]["direct"]["invalid_request_rate"])
        self.assertIsNone(report["per_task"]["direct"]["execution_failure_rate"])
        self.assertEqual(report["overall"]["direct_unnecessary_tool_episode_rate"], 0)
        self.assertIsNone(report["per_task"]["compose"]["direct_unnecessary_tool_episode_rate"])

    def test_direct_unnecessary_calls_are_episode_based(self):
        rows = [trace("a", True, 0, task="direct"), trace("b", True, 2, task="direct"),
                trace("c", False, 1, task="direct"), trace("d", True, 0)]
        report = summarize(rows)
        self.assertEqual(report["overall"]["direct_episodes"], 3)
        self.assertEqual(report["overall"]["direct_unnecessary_tool_episode_rate"], 2 / 3)
        self.assertEqual(report["per_task"]["direct"]["direct_unnecessary_tool_episode_rate"], 2 / 3)

    def test_zero_denominator_rates_and_unavailable_tokens_are_na(self):
        row = trace("a", True, 0, task="direct")
        row["metrics"].update(generated_tokens=0, generation_token_count_available=False)
        report = summarize([row])
        self.assertIsNone(report["overall"]["invalid_request_rate"])
        self.assertIsNone(report["overall"]["execution_failure_rate"])
        self.assertIsNone(report["overall"]["mean_generated_tokens"])
        self.assertEqual(report["overall"]["generated_token_coverage"], 0)
        empty = summarize([])["overall"]
        self.assertIsNone(empty["invalid_request_rate"])
        self.assertIsNone(empty["execution_failure_rate"])
        self.assertIsNone(empty["direct_unnecessary_tool_episode_rate"])

    def test_empty_and_unscored_inputs(self):
        self.assertIsNone(summarize([])["overall"]["accuracy"])
        with self.assertRaises(ValueError):
            summarize([{"sample_id": "a", "budget": 4, "task_type": "direct"}])

    def test_paired_comparison_alignment_and_determinism(self):
        left = [trace("a", False, 3), trace("b", True, 2)]
        right = [trace("b", True, 1), trace("a", True, 2)]
        first = paired_bootstrap(left, right, iterations=200, seed=3)
        self.assertEqual(first, paired_bootstrap(left, right, iterations=200, seed=3))
        self.assertEqual(first["deltas"]["accuracy"]["estimate"], 0.5)
        self.assertEqual(first["deltas"]["mean_admitted_calls"]["estimate"], -1)
        self.assertEqual(first["deltas"]["mean_admitted_calls"]["lower"], -1)

    def test_duplicate_missing_mismatched_pairs_rejected(self):
        a = trace("a", True, 1)
        for left, right in (([a, a], [a]), ([a], [trace("b", True, 1)]),
                            ([a], [trace("a", True, 1, task="direct")]),
                            ([a], [trace("a", True, 1, budget=2)]), ([], [])):
            with self.assertRaises(ValueError):
                paired_bootstrap(left, right, iterations=10)

    def test_same_sample_across_budgets_is_one_bootstrap_cluster(self):
        left = [trace("a", False, 2, b) for b in (2, 4)] + [trace("b", True, 2, b) for b in (2, 4)]
        right = [trace("a", True, 1, b) for b in (2, 4)] + [trace("b", True, 1, b) for b in (2, 4)]
        report = paired_bootstrap(left, right, iterations=100)
        self.assertEqual(report["n_pairs"], 4)
        self.assertEqual(report["n_samples"], 2)
        self.assertEqual(report["strata"], [{"task_type": "compose", "budgets": [2, 4], "samples": 2}])

    def test_known_paired_zero_delta(self):
        rows = [trace("a", True, 1, task="direct"), trace("b", False, 3)]
        result = paired_bootstrap(rows, list(reversed(rows)), iterations=50)
        for metric in result["deltas"].values():
            self.assertEqual(metric, {"estimate": 0, "lower": 0, "upper": 0})


if __name__ == "__main__":
    unittest.main()
