"""CPU-only integration checks. Stub HTTP responses are never model evidence."""
import json
from pathlib import Path

import pytest

from tool_r1.cli import main
from tool_r1.data import audit_dataset, generate_dataset, load_jsonl
from tool_r1.experiments import (
    NO_TOOL_SYSTEM, _model_provenance, compare_runs, digest, evaluate, load_run,
    scripted_smoke, write_json,
)
from tool_r1.rewards import verify_answer


@pytest.fixture
def tiny_data(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(directory, train=8, validation=8, test=8, seed=42)
    return directory


@pytest.fixture
def stub_http(monkeypatch):
    """Exercise real HTTPPolicy encoding while replacing only network transport."""
    requests = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            # Fixed answer independent of sample/target: an interface stub only.
            text = '{"action":"final","answer":{"type":"number","value":6,"unit":"boxes"}}'
            return json.dumps({"choices": [{"message": {"content": text}}],
                               "usage": {"completion_tokens": 19}}).encode()[:size]
    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", respond)
    return requests


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fake_export(root, method, sft, *, seed=42):
    """Tiny hashable artifacts for metadata tests; these are not model weights."""
    run = root / method
    model = run / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"test_only":true}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test-only-not-a-model-" + method.encode())
    exported = {"stage": "grpo_export", "merged": True, "merge_base": "/same/sft/base",
                "merged_files": {name: digest(model / name) for name in ("config.json", "model.safetensors")}}
    write_json(model / "tool_r1_export.json", exported)
    write_json(model / "tool_r1_checkpoint.json", {"run": str(run), "step": 20, "selection": "explicit"})
    from tool_r1.training.launch import VERL_COMMIT, build_overrides
    config = read_json(Path(__file__).resolve().parents[1] / "configs/a6000.json")
    overrides, agent = build_overrides(config, model=root / "shared-sft", data=root / "data",
                                      prepared=root / "prepared", run_dir=run, mode=method, seed=seed)
    write_json(run / "run.json", {"args": {"mode": method, "seed": seed, "steps": None, "smoke": False,
                                          "output": str(run), "resume": False},
                                 "sft": sft, "capacity_test": False, "config": config,
                                 "command": ["python", "-m", "verl.trainer.main_ppo", *overrides],
                                 "agent": agent, "verl_commit": VERL_COMMIT,
                                 "dataset_sha256": sft.get("dataset_sha256", "fixed-data")})
    return model


def test_cli_generate_audit_smoke_and_report_form_a_complete_pipeline(tmp_path, capsys):
    data = tmp_path / "data"
    output = tmp_path / "smoke"
    main(["generate", "--data", str(data), "--train", "8", "--validation", "8", "--test", "8"])
    generated = json.loads(capsys.readouterr().out)
    assert generated["splits"]["train"]["samples"] == 8
    main(["audit", "--data", str(data)])
    assert json.loads(capsys.readouterr().out)["ok"]
    main(["smoke", "--data", str(data), "--output", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["evaluation_kind"] == "scripted_smoke"
    manifest, traces = load_run(output)
    summary = read_json(output / "summary.json")
    assert manifest["status"] == "complete"
    assert manifest["method"] == "teacher_replay"
    assert manifest["episodes"] == summary["overall"]["n"] == len(traces) == 8
    assert manifest["traces_sha256"] == digest(output / "traces.jsonl")
    assert all(trace["score"]["correctness"] and trace["evidence_covered"] for trace in traces)
    assert summary["overall"]["accuracy"] == 1
    # Smoke run replays training demonstrations. The exact task subset depends
    # on the train-size rounding; require the four atomic types plus at least
    # one intermediate skill to prove the new schema is exercised end to end.
    assert {"direct", "retrieve", "compute", "compose"}.issubset(set(summary["per_task"]))
    assert "不是语言模型实验" in (output / "report.md").read_text(encoding="utf-8")
    main(["report", "--run", str(output)])
    assert json.loads(capsys.readouterr().out)["overall"] == summary["overall"]
    with pytest.raises(FileExistsError):
        scripted_smoke(data, output)


def test_http_evaluation_records_prompts_scores_usage_and_frozen_artifacts(tiny_data, tmp_path, stub_http):
    output = tmp_path / "evaluation"
    result = evaluate(tiny_data, output, method="M1", model_source="unit-test/base",
                      budgets=(1, 3), limit=4, seed=123)
    manifest, traces = load_run(output)
    summary = read_json(output / "summary.json")
    samples = {sample["id"]: sample for sample in load_jsonl(tiny_data / "validation/samples.jsonl")}
    assert result["overall"] == summary["overall"]
    assert manifest["episodes"] == len(traces) == len(stub_http) == 8
    assert manifest["budgets"] == [1, 3]
    assert manifest["split"] == "validation"
    assert manifest["dataset_sha256"] == audit_dataset(tiny_data)["dataset_sha256"]
    for payload, trace in zip(stub_http, traces):
        sample = samples[trace["sample_id"]]
        assert payload["messages"] == trace["messages"][:2]
        assert payload["messages"][1] == {"role": "user", "content": sample["question"]}
        serialized = json.dumps(payload["messages"])
        for metadata_key in ("abs_tolerance", "rel_tolerance", "source_doc_ids", "template_family", "task_type"):
            assert metadata_key not in serialized
        assert sample["id"] not in serialized
        assert all(source_id not in serialized for source_id in sample["source_doc_ids"])
        assert trace["score"]["correctness"] is verify_answer(trace["answer"], sample["answer"])
        assert trace["metrics"]["generated_tokens"] == 19
        assert trace["metrics"]["admitted_calls"] == 0
    assert summary["evaluation_kinds"] == {"language_model": 8}
    assert summary["overall"]["invalid_request_rate"] is None
    assert "语言模型评测；数据为合成任务" in (output / "report.md").read_text(encoding="utf-8")


def test_m0_transmits_and_saves_actual_no_tool_system_with_budget_zero(tiny_data, tmp_path, stub_http):
    output = tmp_path / "M0"
    evaluate(tiny_data, output, method="M0", model_source="unit-test/base", budgets=(2, 4, 6), limit=2)
    manifest, traces = load_run(output)
    assert manifest["budgets"] == [0]
    assert len(traces) == 2
    for payload, trace in zip(stub_http, traces):
        assert trace["budget"] == 0
        assert payload["messages"][0] == trace["messages"][0] == {"role": "system", "content": NO_TOOL_SYSTEM}
        assert "No tools are available" in trace["messages"][0]["content"]
        assert "Tool request:" not in trace["messages"][0]["content"]
        assert trace["metrics"]["admitted_calls"] == 0


def test_generation_failure_keeps_partial_evidence_but_cannot_be_completed_or_reported(tiny_data, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("secret-token=should-never-appear")
    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Model generation failed"):
        evaluate(tiny_data, output, method="M1", model_source="unit-test/base", limit=4)
    manifest = read_json(output / "manifest.json")
    traces = load_jsonl(output / "traces.jsonl")
    assert manifest["status"] == "failed"
    assert manifest["reason"] == "model_generation_failed"
    assert manifest["completed_episodes"] == len(traces) == 1
    assert traces[0]["terminal_reason"] == "generation_error"
    assert "should-never-appear" not in (output / "traces.jsonl").read_text(encoding="utf-8")
    assert not (output / "summary.json").exists()
    assert not (output / "report.md").exists()
    with pytest.raises(ValueError, match="Only completed"):
        load_run(output)


def test_completed_trace_tampering_is_rejected(tiny_data, tmp_path, stub_http):
    output = tmp_path / "run"
    evaluate(tiny_data, output, method="M1", model_source="unit-test/base", limit=2)
    with (output / "traces.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="traces changed"):
        load_run(output)
    with pytest.raises(ValueError, match="traces changed"):
        compare_runs(output, output, iterations=10)


def test_compare_rejects_scripted_results_and_mismatched_evaluation_provenance(tiny_data, tmp_path, stub_http):
    smoke = tmp_path / "smoke"
    scripted_smoke(tiny_data, smoke)
    with pytest.raises(ValueError, match="genuine language-model"):
        compare_runs(smoke, smoke, iterations=10)
    left, right = tmp_path / "left", tmp_path / "right"
    evaluate(tiny_data, left, method="M1", model_source="unit-test/base", limit=4, seed=42)
    evaluate(tiny_data, right, method="M1", model_source="unit-test/base", limit=4, seed=99)
    with pytest.raises(ValueError, match="evaluation_seed"):
        compare_runs(left, right, iterations=10)


def test_sft_provenance_requires_metadata_and_checks_merged_files(tmp_path):
    model = tmp_path / "sft"
    model.mkdir()
    with pytest.raises(ValueError, match="tool_r1_sft.json"):
        _model_provenance(str(model), "M2")
    (model / "model.safetensors").write_bytes(b"test-only-sft")
    metadata = {"stage": "sft", "merged_files": {"model.safetensors": digest(model / "model.safetensors")}}
    write_json(model / "tool_r1_sft.json", metadata)
    assert _model_provenance(str(model), "M2")["sft"] == metadata
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed or missing"):
        _model_provenance(str(model), "M2")


def test_grpo_reads_export_metadata_and_original_sft_from_training_run(tmp_path):
    sft = {"stage": "sft", "dataset_sha256": "fixed-data", "adapter_sha256": "fixed-sft"}
    model = fake_export(tmp_path, "M4", sft)
    assert not (model / "tool_r1_sft.json").exists()
    provenance = _model_provenance(str(model), "M4")
    assert provenance["sft"] == sft
    assert provenance["export"]["stage"] == "grpo_export"
    assert provenance["checkpoint"]["step"] == 20
    assert provenance["training_seed"] == 42
    assert not provenance["capacity_test"]
    assert provenance["training_run_sha256"] == digest(model.parent / "run.json")
    with pytest.raises(ValueError, match="training mode"):
        _model_provenance(str(model), "M3")
    (model / "model.safetensors").write_bytes(b"altered")
    with pytest.raises(ValueError, match="changed or missing"):
        _model_provenance(str(model), "M4")


def test_rl_ablation_accepts_shared_sft_and_rejects_different_initialization(tiny_data, tmp_path, stub_http):
    sft = {"stage": "sft", "dataset_sha256": audit_dataset(tiny_data)["dataset_sha256"], "adapter_sha256": "same-sft"}
    left_model, right_model = (fake_export(tmp_path / "models", mode, sft) for mode in ("M3", "M4"))
    left, right = tmp_path / "left", tmp_path / "right"
    evaluate(tiny_data, left, method="M3", model_source=str(left_model), limit=4)
    evaluate(tiny_data, right, method="M4", model_source=str(right_model), limit=4)
    comparison = compare_runs(left, right, iterations=20)
    assert comparison["accuracy_guardrail_passed"]
    assert not comparison["efficiency_supported"]
    assert comparison["left_method"] == "M3" and comparison["right_method"] == "M4"
    manifest = read_json(right / "manifest.json")
    manifest["model"]["sft"]["adapter_sha256"] = "different-sft"
    write_json(right / "manifest.json", manifest)
    with pytest.raises(ValueError, match="RL ablation differs in sft"):
        compare_runs(left, right, iterations=20)


def test_model_dataset_mismatch_is_rejected_before_output_creation(tiny_data, tmp_path, stub_http):
    model = fake_export(tmp_path / "models", "M4", {"dataset_sha256": "different-dataset"})
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="dataset fingerprints differ"):
        evaluate(tiny_data, output, method="M4", model_source=str(model))
    assert not output.exists()
    assert not stub_http


@pytest.mark.parametrize("method", ["M0", "M1"])
def test_base_methods_cannot_be_labeled_with_trained_tool_r1_exports(tmp_path, method):
    model = fake_export(tmp_path, "M4", {"dataset_sha256": "fixed-data"})
    with pytest.raises(ValueError, match="untrained base model"):
        _model_provenance(str(model), method)


@pytest.mark.parametrize("change", ["config", "effective_override", "checkpoint_step", "verl_commit", "steps"])
def test_rl_ablation_rejects_training_confounders(tiny_data, tmp_path, stub_http, change):
    sft = {"stage": "sft", "dataset_sha256": audit_dataset(tiny_data)["dataset_sha256"], "adapter_sha256": "same-sft"}
    left_model, right_model = (fake_export(tmp_path / "models", mode, sft) for mode in ("M3", "M4"))
    run_path = right_model.parent / "run.json"
    run = read_json(run_path)
    if change == "config":
        run["config"]["learning_rate"] *= 2
    elif change == "effective_override":
        run["command"] = ["actor_rollout_ref.actor.optim.lr=0.5" if item.startswith("actor_rollout_ref.actor.optim.lr=") else item for item in run["command"]]
    elif change == "checkpoint_step":
        checkpoint_path = right_model / "tool_r1_checkpoint.json"
        checkpoint = read_json(checkpoint_path)
        checkpoint["step"] = 40
        write_json(checkpoint_path, checkpoint)
    elif change == "verl_commit":
        run["verl_commit"] = "different-upstream-commit"
    elif change == "steps":
        run["args"]["steps"] = 500
    write_json(run_path, run)
    left, right = tmp_path / "left", tmp_path / "right"
    evaluate(tiny_data, left, method="M3", model_source=str(left_model), limit=4)
    evaluate(tiny_data, right, method="M4", model_source=str(right_model), limit=4)
    with pytest.raises(ValueError, match="RL ablation differs in training_control"):
        compare_runs(left, right, iterations=10)


def test_budget_visibility_treatment_remains_a_valid_controlled_ablation(tiny_data, tmp_path, stub_http):
    sft = {"stage": "sft", "dataset_sha256": audit_dataset(tiny_data)["dataset_sha256"], "adapter_sha256": "same-sft"}
    left_model, right_model = (fake_export(tmp_path / "models", mode, sft) for mode in ("M4", "M5"))
    left, right = tmp_path / "visible", tmp_path / "hidden"
    evaluate(tiny_data, left, method="M4", model_source=str(left_model), limit=4)
    evaluate(tiny_data, right, method="M5", model_source=str(right_model), limit=4)
    result = compare_runs(left, right, iterations=10)
    assert result["accuracy_guardrail_passed"]
    _, traces = load_run(right)
    assert all("Total tool credits:" not in trace["messages"][0]["content"] for trace in traces)


def test_grpo_rejects_effective_reward_weight_that_disagrees_with_method(tmp_path):
    model = fake_export(tmp_path, "M3", {"dataset_sha256": "fixed-data"})
    path = model.parent / "run.json"
    run = read_json(path)
    run["agent"][0]["cost_weight"] = 0.3
    write_json(path, run)
    with pytest.raises(ValueError, match="reward weight differs"):
        _model_provenance(str(model), "M3")


def test_grpo_provenance_requires_effective_training_settings(tmp_path):
    model = fake_export(tmp_path, "M4", {"dataset_sha256": "fixed-data"})
    path = model.parent / "run.json"
    run = read_json(path)
    del run["command"]
    write_json(path, run)
    with pytest.raises(ValueError, match="nonempty command"):
        _model_provenance(str(model), "M4")
