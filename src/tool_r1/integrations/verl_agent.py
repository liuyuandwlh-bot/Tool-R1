"""verl v0.7.0 AgentLoop adapter, token-in/token-out with explicit tool masks."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from tool_r1.data import load_jsonl
from tool_r1.runner import Environment
from tool_r1.tools.retrieval import BM25
from tool_r1.training.chat import append_segment, observation_tokens, prompt_tokens, validate_chatml
from .reward import score_episode


@lru_cache(maxsize=8)
def _corpus(path: str):
    return BM25(load_jsonl(path))


class ToolR1AgentLoop(AgentLoopBase):
    def __init__(self, *args, data_root="data", cost_weight=0.1, cost_reference=6,
                 max_action_tokens=384, python_backend="restricted", trace_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        validate_chatml(self.tokenizer)
        self.data_root = Path(data_root).resolve()
        self.cost_weight, self.cost_reference = float(cost_weight), int(cost_reference)
        self.max_action_tokens, self.python_backend = int(max_action_tokens), python_backend
        self.trace_dir = trace_dir

    async def run(self, sampling_params, **kwargs):
        extra = kwargs["extra_info"]
        split = extra["split"]
        if split not in {"train", "validation"}:
            raise ValueError("Training AgentLoop cannot read held-out test data")
        docs = _corpus(str(self.data_root / split / "corpus.jsonl"))
        env = Environment(docs, budget=int(extra["budget"]), budget_visible=bool(extra["budget_visible"]),
                          python_backend=self.python_backend)
        messages = env.initial_messages(extra["question"])
        # Auditable prompt equality: metadata and reward targets are not model input.
        raw = [dict(message) for message in kwargs["raw_prompt"]]
        if raw != messages:
            raise ValueError("Exported prompt differs from Environment; re-run training.prepare")
        initial_ids = prompt_tokens(self.tokenizer, messages)
        rollout = self.config.actor_rollout_ref.rollout
        if len(initial_ids) > int(rollout.prompt_length):
            raise ValueError("Initial prompt exceeds configured length")
        response_ids, response_mask, logprobs, steps = [], [], [], []
        generation_time, tool_time, turns = 0.0, 0.0, 0
        terminal, answer, truncated = "turn_limit", None, False
        request_id = uuid4().hex
        max_response = int(rollout.response_length)
        max_context = int(rollout.max_model_len)
        for _ in range(env.budget + 2):
            remaining = min(max_response - len(response_ids), max_context - len(initial_ids) - len(response_ids))
            if remaining <= 0:
                terminal, truncated = "context_limit", True
                break
            params = dict(sampling_params)
            params["max_tokens"] = min(self.max_action_tokens, remaining)
            start = time.perf_counter()
            output = await self.server_manager.generate(request_id=request_id,
                        prompt_ids=initial_ids + response_ids, sampling_params=params)
            generation_time += time.perf_counter() - start
            generated = list(output.token_ids)
            if not generated or len(generated) > remaining:
                raise ValueError("Generation server returned empty or overlength tokens")
            append_segment(response_ids, response_mask, generated, generated=True)
            if sampling_params.get("logprobs"):
                if output.log_probs is None or len(output.log_probs) != len(generated):
                    raise ValueError("Missing rollout token log probabilities")
                logprobs.extend(output.log_probs)
            turns += 1
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            # Never execute truncated code or reward a clipped JSON prefix.
            if generated[-1] != self.tokenizer.eos_token_id or getattr(output, "stop_reason", None) == "aborted":
                terminal, truncated = "action_token_limit", True
                break
            start = time.perf_counter()
            result = await asyncio.to_thread(env.step, text)
            tool_time += time.perf_counter() - start
            steps.append({"action_text": text, **result})
            if result["done"]:
                answer, terminal = result.get("answer"), result.get("terminal_reason", "done")
                break
            observation = result.get("observation")
            if not isinstance(observation, str):
                raise ValueError("Nonterminal environment step must produce an observation")
            context = observation_tokens(self.tokenizer, observation)
            # Leave room for at least one next generated token; do not end on tool text.
            if len(response_ids) + len(context) >= max_response or len(initial_ids) + len(response_ids) + len(context) >= max_context:
                terminal, truncated = "observation_context_limit", True
                break
            append_segment(response_ids, response_mask, context, generated=False)
            if logprobs:
                logprobs.extend([0.0] * len(context))
            messages.append({"role": "user", "content": observation})
        reward = score_episode(answer, kwargs["reward_model"]["ground_truth"], env.metrics,
                               cost_weight=self.cost_weight, cost_reference=self.cost_reference, truncated=truncated)
        trace = {"sample_id": extra["sample_id"], "task_type": extra["task_type"], "split": split,
                 "budget": int(extra["budget"]), "budget_visible": bool(extra["budget_visible"]),
                 "request_id": request_id, "answer": answer, "messages": messages, "steps": steps,
                 "metrics": dict(env.metrics), "terminal_reason": terminal, "reward": reward,
                 "generated_tokens": sum(response_mask), "context_tokens": len(initial_ids) + len(response_ids)}
        if self.trace_dir:
            directory = Path(self.trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            # Each process writes a separate file; each line is one complete trajectory.
            with (directory / f"worker-{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return AgentLoopOutput(prompt_ids=initial_ids, response_ids=response_ids, response_mask=response_mask,
                    response_logprobs=logprobs if sampling_params.get("logprobs") else None,
                    num_turns=len(messages), reward_score=reward["score"],
                    metrics={"generate_sequences": generation_time, "tool_calls": tool_time},
                    extra_fields={"reward_extra_info": reward, "tool_r1_trace": trace})
