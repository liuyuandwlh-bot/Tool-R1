# Tool-R1 implementation contract

This file records shared interfaces for implementation. Python package root is `src/tool_r1`. Runtime core must work on Python 3.10+ with the standard library. Optional ML dependencies are imported lazily.

## Data

Samples are JSON objects with `id`, `question`, `task_type` (direct/retrieve/compute/compose), `split` (train/validation/test), `template_family`, `source_doc_ids`, and `answer` ({type: number/text/set/json, value: ..., unit: ...}). Additional provenance fields are allowed. Samples never carry their answer into model prompts. Documents are {id, text, split, entity}. Files: data/<split>/samples.jsonl and corpus.jsonl. Training-only demonstrations may be stored separately.

## Actions

The model produces ONE strict JSON object per assistant turn:

    {"action":"tool","name":"retrieve","arguments":{"query":"...","top_k":3}}
    {"action":"tool","name":"python","arguments":{"code":"print(sum([1,2,3]))"}}
    {"action":"final","answer":{"type":"number","value":6,"unit":"count"}}

No tool metadata, sample answer, task_type or evidence is inserted into the question. Tool observations use a user message containing an explicitly generated environment JSON object, because this JSON-action protocol does not use native function-call messages. Observations are never parsed as actions/final answers.

## Root-owned interfaces

- `protocol.parse_action(text) -> dict`, raises ValueError on invalid action/schema.
- `runner.Environment(documents, budget=4, budget_visible=True, python_backend="restricted")`; `.initial_messages(question) -> list[dict]`; `.step(action_text) -> dict` with `done`, `observation` (string or None), `answer` (dict or None), `terminal_reason`; `.metrics` dict (requests, admitted_calls, executed_calls, invalid_requests, execution_failures, budget_violations).
- `.step` accepts final or tool requests. Each accepted tool attempt, including invalid arguments/name/protocol, spends one available call credit. A tool request with no credit is refused and sets final-only mode. A second non-final action in final-only mode terminates. Final action never spends credit. Max turns is budget+2.
- `runner.run_episode(sample, policy, documents, budget=4, budget_visible=True, python_backend="restricted", **kwargs) -> dict`; policy has `.generate(messages) -> str`; returned trace includes sample_id/task_type/budget/budget_visible/answer/messages/steps/metrics/terminal_reason. Runtime excludes sample ground truth from policy context. Scoring is appended outside runner.
- `tools.retrieval.BM25(documents).search(query, top_k=3) -> list[dict]`.

## Agent-owned interfaces

- Data/eval agent: `data.generate_dataset(output_dir, train=2400, validation=400, test=800, seed=42) -> dict`; `data.audit_dataset(output_dir) -> dict`; `rewards.verify_answer(prediction, target) -> bool`; `rewards.score_answer(prediction, target, admitted_calls, cost_weight=0.1, cost_reference=6) -> dict` including correctness/reward/cost; `evaluation.summarize(traces) -> dict`; paired comparison helper. CLI imports these; core evaluation uses only stdlib. Reports must distinguish scripted checks from language-model metrics.
- Sandbox agent: `tools.python_tool.run_python(code, backend="restricted", timeout=5.0, max_output_chars=4096) -> dict` returning {ok, result: stdout string, error: string or None, backend}; provide production Docker option and safe deterministic restricted-Python interpreter usable without Docker. No host eval/exec of model-generated code. Add module-specific tests.
- Training agent: owns `training/`, `integrations/`, `configs/`, `scripts/` (except root-owned preflight/package scripts). Implement actual pinned verl AgentLoop adapter and SFT / GRPO preparation-launch recipes for single RTX A6000 48GB; import Environment and reward contract above. Do not invent benchmark results or claim GPU validation on this Windows machine.

All scripts resolve repository paths safely. Tests focus on correctness, masks, budgets, leakage and execution isolation. No purchases, remote execution, credential changes or model downloads during implementation without an explicit scoped decision by root.
