"""Token bookkeeping for the supported Qwen2.5 ChatML tokenizer.

Keep sampled token IDs unchanged. Environment messages and assistant headers
are context (mask=0); only tokens actually sampled by the policy have mask=1.
Fail explicitly on a different template rather than train with incorrect masks.
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def validate_chatml(tokenizer) -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    expected = f"{IM_START}system\ns{IM_END}\n{IM_START}user\nu{IM_END}\n{IM_START}assistant\n"
    actual = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if actual != expected or tokenizer.encode(IM_END, add_special_tokens=False) != [tokenizer.eos_token_id]:
        raise ValueError("This recipe requires the unmodified Qwen2.5-Instruct ChatML template and EOS.")


def prompt_tokens(tokenizer, messages: list[dict]) -> list[int]:
    return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def observation_tokens(tokenizer, observation: str) -> list[int]:
    # Generation ended with IM_END; the template's trailing newline is context.
    text = f"\n{IM_START}user\n{observation}{IM_END}\n{IM_START}assistant\n"
    return list(tokenizer.encode(text, add_special_tokens=False))


def append_segment(ids: list[int], mask: list[int], tokens: list[int], *, generated: bool) -> None:
    if len(ids) != len(mask):
        raise ValueError("Response IDs/mask length mismatch")
    ids.extend(tokens)
    mask.extend([int(generated)] * len(tokens))


def sft_turn(tokenizer, messages: list[dict], action: str, max_length: int) -> dict:
    """One teacher turn with full previous context and assistant-only labels.

All context (including previous assistant turns) is ignored in this example.
Every teacher action appears once as the target of its own training example.
No right truncation is allowed to silently delete the final-answer supervision.
"""
    prefix = prompt_tokens(tokenizer, messages)
    target = list(tokenizer.encode(action, add_special_tokens=False)) + [tokenizer.eos_token_id]
    ids = prefix + target
    if len(ids) > max_length:
        raise ValueError(f"SFT turn has {len(ids)} tokens > max_length={max_length}; increase context explicitly")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + target}
