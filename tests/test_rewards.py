import math
import unittest

from tool_r1.rewards import score_answer, verify_answer


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.target = {"type": "number", "value": 100, "unit": "boxes", "abs_tolerance": "0.01", "rel_tolerance": "0"}

    def number(self, value, unit="boxes"):
        return {"type": "number", "value": value, "unit": unit}

    def test_numeric_tolerance_boundary_and_units(self):
        self.assertTrue(verify_answer(self.number("100.01"), self.target))
        self.assertFalse(verify_answer(self.number("100.010001"), self.target))
        self.assertFalse(verify_answer(self.number(100, "credits"), self.target))
        self.assertFalse(verify_answer({"type": "number", "value": 100}, self.target))
        self.assertTrue(verify_answer(self.number(100, " BOXES "), self.target))

    def test_relative_tolerance_and_explicit_alias(self):
        target = {**self.target, "value": 1000, "rel_tolerance": "0.001", "abs_tolerance": "0", "unit_aliases": ["box"]}
        self.assertTrue(verify_answer(self.number(1001, "box"), target))
        self.assertFalse(verify_answer(self.number("1001.001"), target))

    def test_reject_boolean_nonfinite_and_embedded_units(self):
        for value in (True, False, math.nan, math.inf, -math.inf, "NaN", "Infinity", "100 boxes", [], "1e999999"):
            self.assertFalse(verify_answer(self.number(value), self.target), repr(value))
        self.assertFalse(verify_answer(self.number(1), {**self.target, "value": True}))
        self.assertFalse(verify_answer(None, self.target))

    def test_text_alias_normalization(self):
        target = {"type": "text", "value": "New York", "aliases": ["NYC"]}
        self.assertTrue(verify_answer({"type": "text", "value": "  new   YORK "}, target))
        self.assertTrue(verify_answer({"type": "text", "value": "nyc"}, target))
        self.assertFalse(verify_answer({"type": "text", "value": "York"}, target))

    def test_set_and_json_semantics(self):
        target = {"type": "set", "value": ["Alpha", "beta"]}
        self.assertTrue(verify_answer({"type": "set", "value": ["BETA", "alpha", "alpha"]}, target))
        self.assertFalse(verify_answer({"type": "set", "value": ["alpha"]}, target))
        self.assertFalse(verify_answer({"type": "set", "value": [True]}, {"type": "set", "value": [1]}))
        self.assertFalse(verify_answer({"type": "json", "value": {"a": True}}, {"type": "json", "value": {"a": 1}}))
        self.assertTrue(verify_answer({"type": "json", "value": {"b": 2, "a": 1}}, {"type": "json", "value": {"a": 1, "b": 2}}))

    def test_correctness_gating_and_fixed_cost(self):
        correct = score_answer(self.number(100), self.target, 3, 0.1, 6)
        wrong = score_answer(self.number(101), self.target, 0, 0.1, 6)
        self.assertEqual(correct["reward"], 0.95)
        self.assertEqual(correct["cost"], 0.5)
        self.assertEqual(wrong["reward"], 0)
        self.assertEqual(score_answer(self.number(100), self.target, 100)["reward"], 0.9)
        self.assertEqual(score_answer(self.number(100), self.target, 3, cost_weight=0)["reward"], 1)

    def test_invalid_reward_parameters(self):
        for args in ({"admitted_calls": -1}, {"admitted_calls": True}, {"admitted_calls": 1, "cost_weight": 1},
                     {"admitted_calls": 1, "cost_weight": math.nan}, {"admitted_calls": 1, "cost_reference": 0}):
            with self.assertRaises(ValueError):
                score_answer(self.number(100), self.target, **args)


if __name__ == "__main__":
    unittest.main()
