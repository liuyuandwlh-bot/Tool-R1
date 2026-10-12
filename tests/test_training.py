import json
from pathlib import Path
import unittest

from tool_r1.integrations.reward import compute_score, score_episode
from tool_r1.training.chat import append_segment, observation_tokens, prompt_tokens, sft_turn, validate_chatml
from tool_r1.training.launch import build_overrides


class ChatMLTokenizer:
    eos_token_id = 999999
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith("<|im_end|>"):
                result.append(self.eos_token_id)
                text = text[len("<|im_end|>"):]
            else:
                result.append(ord(text[0]) + 1)
                text = text[1:]
        return result

    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else "<|im_end|>") if x == self.eos_token_id else chr(x - 1) for x in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


class TrainingTests(unittest.TestCase):
    def test_incremental_chatml_matches_complete_template(self):
        tokenizer = ChatMLTokenizer()
        validate_chatml(tokenizer)
        initial = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "question"}]
        action = '{"action":"tool","name":"retrieve","arguments":{"query":"x"}}'
        observation = '{"environment":{"result":"source"}}'
        sampled = tokenizer.encode(action) + [tokenizer.eos_token_id]
        continuation = observation_tokens(tokenizer, observation)
        assembled = prompt_tokens(tokenizer, initial) + sampled + continuation
        complete = prompt_tokens(tokenizer, initial + [{"role": "assistant", "content": action}, {"role": "user", "content": observation}])
        self.assertEqual(assembled, complete)
        ids, mask = [], []
        append_segment(ids, mask, sampled, generated=True)
        append_segment(ids, mask, continuation, generated=False)
        self.assertEqual(mask, [1] * len(sampled) + [0] * len(continuation))

    def test_sft_context_has_no_labels_and_no_silent_truncation(self):
        tokenizer = ChatMLTokenizer()
        messages = [{"role": "system", "content": "SECRET_CONTEXT"}, {"role": "user", "content": "observation"}]
        row = sft_turn(tokenizer, messages, '{"action":"final","answer":7}', 4096)
        prefix = len(prompt_tokens(tokenizer, messages))
        self.assertEqual(row["labels"][:prefix], [-100] * prefix)
        self.assertEqual(row["labels"][prefix:], row["input_ids"][prefix:])
        self.assertEqual(row["labels"][-1], tokenizer.eos_token_id)
        with self.assertRaises(ValueError):
            sft_turn(tokenizer, messages, "long target", 5)

    def test_reward_uses_trusted_answer_and_fixed_cost(self):
        target = {"type": "number", "value": 7, "unit": "boxes"}
        metrics = {"admitted_calls": 2, "requests": 2}
        score = score_episode(target, json.dumps(target), metrics, cost_weight=0.1)
        self.assertAlmostEqual(score["score"], 1 - .1 * 2 / 6)
        self.assertEqual(score_episode(None, target, metrics)["score"], 0)
        self.assertEqual(score_episode(target, target, metrics, truncated=True)["score"], 0)
        with self.assertRaises(RuntimeError):
            compute_score("tool_r1", json.dumps(target), target)

    def test_modes_share_training_settings_and_sft_base(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/a6000.json").read_text())
        args = dict(model=Path("sft/merged"), data=Path("data"), prepared=Path("data/prepared"), run_dir=Path("out"))
        a3, g3 = build_overrides(config, **args, mode="M3")
        a4, g4 = build_overrides(config, **args, mode="M4")
        a5, g5 = build_overrides(config, **args, mode="M5")
        self.assertEqual(a3, a4)
        self.assertEqual(g3[0]["cost_weight"], 0)
        self.assertEqual(g4[0]["cost_weight"], .1)
        self.assertEqual(g5[0]["cost_weight"], .1)
        self.assertEqual([x for x in a4 if not x.startswith(("data.train_files", "data.val_files"))],
                         [x for x in a5 if not x.startswith(("data.train_files", "data.val_files"))])
        self.assertIn('actor_rollout_ref.rollout.n=4', a4)
        self.assertIn('actor_rollout_ref.model.path=' + json.dumps(str(Path('sft/merged'))), a4)


if __name__ == "__main__":
    unittest.main()
