import json
import tempfile
import unittest
from pathlib import Path

from tool_r1.data import audit_dataset, generate_dataset, load_jsonl


class DataTests(unittest.TestCase):
    def test_determinism_split_isolation_and_heldout_wording(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = generate_dataset(a, train=24, validation=12, test=24, seed=7)
            second = generate_dataset(b, train=24, validation=12, test=24, seed=7)
            self.assertEqual(first, second)
            self.assertTrue(audit_dataset(a)["ok"])
            seen_entities, seen_sources = set(), set()
            for split in ("train", "validation", "test"):
                rows = load_jsonl(Path(a) / split / "samples.jsonl")
                entities = {row["metadata"]["entity"] for row in rows}
                sources = {source for row in rows for source in row["source_doc_ids"]}
                self.assertFalse(entities & seen_entities)
                self.assertFalse(sources & seen_sources)
                seen_entities.update(entities)
                seen_sources.update(sources)
                self.assertTrue(all("reference_actions" not in row for row in rows))
                statuses = {row["template_status"] for row in rows}
                self.assertEqual(statuses, {"known", "heldout"} if split == "test" else {"known"})
            demonstrations = load_jsonl(Path(a) / "train" / "demonstrations.jsonl")
            self.assertEqual(len(demonstrations), 24)
            self.assertFalse((Path(a) / "test" / "demonstrations.jsonl").exists())

    def test_counts_task_balance_and_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_dataset(directory, train=40, validation=8, test=16)
            train_tasks = manifest["splits"]["train"]["tasks"]
            self.assertEqual(set(train_tasks), {"direct", "retrieve", "compute", "compose",
                                                "multi_hop_2", "conditional", "conflict", "no_tool"})
            self.assertEqual(sum(train_tasks.values()), 40)
            # Existing four task types must still appear in the validation split.
            for task in ("direct", "retrieve", "compute", "compose"):
                self.assertGreater(train_tasks[task], 0)
            test_tasks = manifest["splits"]["test"]["tasks"]
            self.assertEqual(set(test_tasks),
                             {"direct", "retrieve", "compute", "compose", "multi_hop_2", "multi_hop_3",
                              "conditional", "conflict", "composed", "no_tool"})
            self.assertEqual(sum(test_tasks.values()), 16)
            corpus = load_jsonl(Path(directory) / "train" / "corpus.jsonl")
            self.assertTrue(any(doc["record"]["status"] == "forecast" for doc in corpus))
            self.assertTrue(all(doc["synthetic"] is True for doc in corpus))
            rows = load_jsonl(Path(directory) / "train" / "samples.jsonl")
            compose = next(row for row in rows if row["task_type"] == "compose")
            self.assertEqual(len(compose["source_doc_ids"]), 2)
            self.assertEqual(compose["answer"]["unit"], "percent")
            self.assertEqual(compose["answer"]["rel_tolerance"], "0")

    def test_tamper_detected_independently_of_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_dataset(directory, train=8, validation=8, test=8)
            path = Path(directory) / "train" / "samples.jsonl"
            rows = load_jsonl(path)
            rows[0]["answer"]["value"] = 999999
            rows[1]["template_family"] = rows[1]["task_type"] + "_v3"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            audit = audit_dataset(directory)
            self.assertFalse(audit["ok"])
            self.assertTrue(any("exact recomputed target" in error for error in audit["errors"]))
            self.assertTrue(any("held-out template leaked" in error for error in audit["errors"]))
            self.assertTrue(any("Hash mismatch" in error for error in audit["errors"]))

    def test_demo_split_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_dataset(directory, train=4, validation=4, test=4)
            (Path(directory) / "test" / "demonstrations.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertFalse(audit_dataset(directory)["ok"])

    def test_source_text_and_structured_facts_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_dataset(directory, train=8, validation=8, test=8)
            path = Path(directory) / "test" / "corpus.jsonl"
            corpus = load_jsonl(path)
            shipment_index = next(index for index, doc in enumerate(corpus)
                                  if doc.get("type", "shipment_actual") in {"shipment_actual", "shipment_forecast"})
            corpus[shipment_index]["text"] = "Synthetic ledger with a corrupted amount: 123."
            path.write_text("\n".join(json.dumps(doc) for doc in corpus) + "\n", encoding="utf-8")
            audit = audit_dataset(directory)
            self.assertFalse(audit["ok"])
            self.assertTrue(any("contradicts its structured facts" in error for error in audit["errors"]))

    def test_zero_and_tiny_splits_and_invalid_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_dataset(directory, train=0, validation=1, test=2)
            self.assertTrue(audit_dataset(directory)["ok"])
            with self.assertRaises(ValueError):
                generate_dataset(directory, train=-1)


if __name__ == "__main__":
    unittest.main()
