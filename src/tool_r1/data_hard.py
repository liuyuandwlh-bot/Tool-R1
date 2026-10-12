"""Hard difficulty dataset extension for GRPO training.

Adds new task types with increased tool-orchestration complexity:
    - multi_hop_4: company -> supplier -> region -> market -> exchange_rate
    - multi_hop_5: multi_hop_4 + validation against historical average
    - chain_of_tools: retrieve -> compute -> conditional -> final answer

These tasks force the model to make multiple correct tool calls in sequence,
and penalize redundant/wasted calls. Combined with adjusted reward function,
this creates gradient signal for cost optimization.
"""

from __future__ import annotations

import hashlib
import json
import random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .data import (
    SCHEMA_VERSION, SPLITS, TEMPLATE_RULES, DOC_TYPES,
    load_jsonl, _dump, _write_rows, _sha256, _heldout_template_index,
    _hex_token, _final, _target, _plan_split_counts,
    _company_supplier_doc, _supplier_region_doc,
)
from .rewards import verify_answer
HARD_TASK_TYPES = ("multi_hop_4", "multi_hop_5", "chain_of_tools")
ALL_TASK_TYPES_WITH_HARD = (
    "direct", "retrieve", "compute", "compose",
    "multi_hop_2", "multi_hop_3", "conditional", "conflict",
    "multi_hop_4", "multi_hop_5", "chain_of_tools", "no_tool",
)

# Templates for hard tasks
HARD_TEMPLATES = {
    "multi_hop_4": [
        "For {entity} in fiscal year {year}: find the top supplier, locate their region, identify that region's major market, and report the exchange rate from {base_currency} to the market's primary currency. Round to 6 decimal places and answer in exchange_rate.",
        "Trace {entity}'s {year} supply chain: top supplier → operating region → primary market → exchange rate from {base_currency}. Round to 6 decimal places, answer in exchange_rate.",
        "Starting from {entity}'s {year} top supplier: determine the region, then the associated market, then the exchange rate (rounded to 6 decimals) from {base_currency} to that market's currency.",
    ],
    "multi_hop_5": [
        "For {entity} in fiscal year {year}, follow: top supplier → region → market → exchange rate (from {base_currency}) → compare against the historical 3-year average. Report the percentage deviation from average, rounded to 2 decimal places, in percent.",
        "{entity}'s {year} supply trace: supplier → region → market → exchange rate from {base_currency}. Then measure how many percent this differs from the 3-year historical average. Round to 2 decimals, answer in percent.",
    ],
    "chain_of_tools": [
        "For {entity} in {year}: retrieve the total transaction volume (in boxes, excluding returns), multiply by the tax rate for their primary supplier's region (rounded to 1 decimal), and compute the estimated tax liability. Answer in credits.",
        "{entity}'s {year} tax burden: get transaction total (boxes, no returns), fetch the region tax rate (1 decimal), multiply. Report estimated tax in credits.",
    ],
}

# Extended task distribution for hard training
HARD_TASK_PLAN = {
    "train-hard-v2": {
        "direct": 0.10, "retrieve": 0.10, "compute": 0.10, "compose": 0.10,
        "multi_hop_2": 0.10, "multi_hop_3": 0.05, "conditional": 0.10, 
        "conflict": 0.05, "multi_hop_4": 0.10, "multi_hop_5": 0.10,
        "chain_of_tools": 0.10, "no_tool": 0.05,
    },
    # v3: Expanded multi_hop_4 from ~10% to ~29% for better GRPO gradient signal
    "train-hard-v3": {
        "direct": 0.088, "retrieve": 0.10, "compute": 0.10, "compose": 0.088,
        "multi_hop_2": 0.088, "multi_hop_3": 0.059, "conditional": 0.088,
        "conflict": 0.035, "multi_hop_4": 0.294, "multi_hop_5": 0.029,
        "chain_of_tools": 0.029, "no_tool": 0.0,
    },
    # v4: Expand chain_of_tools (was 49 → 160) and multi_hop_5 (was 49 → 130)
    # to give GRPO more gradient signal on these weak task families.
    # Keep multi_hop_4 strong (340 samples) but free capacity for the other two.
    "train-hard-v4": {
        "direct": 0.088, "retrieve": 0.10, "compute": 0.10, "compose": 0.088,
        "multi_hop_2": 0.088, "multi_hop_3": 0.059, "conditional": 0.088,
        "conflict": 0.035, "multi_hop_4": 0.20, "multi_hop_5": 0.076,
        "chain_of_tools": 0.094, "no_tool": 0.0,
    },
}


def _make_multi_hop_4(rng: random.Random, seed: int, split: str, index: int, digest: str,
                      heldout: bool) -> tuple[dict, list[dict], dict | None]:
    """4-hop task: company → supplier → region → market → exchange_rate"""
    sample_id = f"{split}-{index:06d}"
    entity = f"NOVA-{digest}"
    year = rng.randrange(2010, 2045)
    supplier_id = f"SUP-{_hex_token(rng, 8)}"
    region_id = f"REG-{_hex_token(rng, 4)}"
    market_id = f"MKT-{_hex_token(rng, 4)}"
    base_currency = rng.choice(["USD", "EUR", "GBP", "JPY"])
    exchange_rate = round(rng.uniform(0.5, 2.5), 6)
    top_spend = rng.randint(300000, 900000)
    
    metadata = {
        "entity": entity, "year": year, "supplier_id": supplier_id,
        "region_id": region_id, "market_id": market_id,
        "base_currency": base_currency, "exchange_rate": exchange_rate,
        "synthetic": True, "generator": "tool_r1.data_hard.v1"
    }
    template_index = _heldout_template_index(rng, heldout)
    docs = []
    source_ids = []
    
    # Company → Supplier (top spend)
    cs_top = _company_supplier_doc(digest, "cs-top", entity, year, supplier_id, top_spend, split, is_top=True)
    docs.append(cs_top)
    source_ids.append(cs_top["id"])
    
    # Distractor suppliers
    for index_d in range(2):
        other_supplier = f"SUP-{_hex_token(rng, 8)}"
        other_spend = rng.randint(100000, top_spend - 50000)
        docs.append(_company_supplier_doc(digest, f"cs-d{index_d}", entity, year,
                                          other_supplier, other_spend, split, is_top=False))
    
    # Supplier → Region
    sr = _supplier_region_doc(digest, "sr", supplier_id, region_id, split, entity)
    docs.append(sr)
    source_ids.append(sr["id"])
    
    # Distractor supplier-regions
    for index_d in range(2):
        other_region = f"REG-{_hex_token(rng, 4)}"
        other_supplier = f"SUP-{_hex_token(rng, 8)}"
        docs.append(_supplier_region_doc(digest, f"sr-d{index_d}", other_supplier, other_region, split, entity))
    
    # Region → Market (new document type)
    rm_doc = {
        "id": f"{digest}:rm", "type": "region_market", "split": split,
        "entity": entity, "synthetic": True,
        "record": {"region_id": region_id, "market_id": market_id},
        "text": f"Region {region_id} profile: serves primary market {market_id}.",
    }
    docs.append(rm_doc)
    source_ids.append(rm_doc["id"])
    
    # Distractor region-markets
    for index_d in range(2):
        other_market = f"MKT-{_hex_token(rng, 4)}"
        other_region = f"REG-{_hex_token(rng, 4)}"
        docs.append({
            "id": f"{digest}:rm-d{index_d}", "type": "region_market", "split": split,
            "entity": entity, "synthetic": True,
            "record": {"region_id": other_region, "market_id": other_market},
            "text": f"Region {other_region} profile: serves primary market {other_market}.",
        })
    
    # Market → Exchange Rate (new document type)
    ex_doc = {
        "id": f"{digest}:ex", "type": "exchange_rate", "split": split,
        "entity": entity, "synthetic": True,
        "record": {"market_id": market_id, "base_currency": base_currency, "rate": exchange_rate},
        "text": f"Market {market_id} trading: {base_currency} exchange rate {exchange_rate:.6f}.",
    }
    docs.append(ex_doc)
    source_ids.append(ex_doc["id"])
    
    # Distractor exchange rates
    for index_d in range(2):
        other_market = f"MKT-{_hex_token(rng, 4)}"
        other_rate = round(rng.uniform(0.5, 2.5), 6)
        docs.append({
            "id": f"{digest}:ex-d{index_d}", "type": "exchange_rate", "split": split,
            "entity": entity, "synthetic": True,
            "record": {"market_id": other_market, "base_currency": base_currency, "rate": other_rate},
            "text": f"Market {other_market} trading: {base_currency} exchange rate {other_rate:.6f}.",
        })
    
    target = _target(exchange_rate, "exchange_rate")
    sample = {
        "id": sample_id,
        "question": HARD_TEMPLATES["multi_hop_4"][template_index].format(**metadata),
        "task_type": "multi_hop_4",
        "split": split,
        "template_family": f"multi_hop_4_v{template_index + 1}",
        "template_status": "heldout" if heldout else "known",
        "source_doc_ids": source_ids,
        "answer": target,
        "metadata": metadata,
        "provenance": "synthetic; no real organizations or financial records"
    }
    
    actions = []
    if split == "train":
        actions = [
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"{entity} {year} top_supplier_id purchase", "top_k": 3}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"supplier {supplier_id} region_id profile", "top_k": 1}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"region {region_id} market_id profile", "top_k": 1}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"market {market_id} exchange_rate {base_currency}", "top_k": 1}},
            _final(target),
        ]
    
    demo = None
    if split == "train":
        demo = {
            "sample_id": sample_id,
            "split": "train",
            "origin": "synthetic deterministic teacher; replay before SFT",
            "reference_actions": actions
        }
    
    return sample, docs, demo


def _make_multi_hop_5(rng: random.Random, seed: int, split: str, index: int, digest: str,
                      heldout: bool) -> tuple[dict, list[dict], dict | None]:
    """5-hop task: multi_hop_4 + compare to 3-year historical average"""
    sample_id = f"{split}-{index:06d}"
    entity = f"NOVA-{digest}"
    year = rng.randrange(2012, 2045)  # Leave room for historical years
    supplier_id = f"SUP-{_hex_token(rng, 8)}"
    region_id = f"REG-{_hex_token(rng, 4)}"
    market_id = f"MKT-{_hex_token(rng, 4)}"
    base_currency = rng.choice(["USD", "EUR", "GBP", "JPY"])
    current_rate = round(rng.uniform(0.5, 2.5), 6)
    avg_rate = round(rng.uniform(0.5, 2.5), 6)
    pct_deviation = round((current_rate - avg_rate) / avg_rate * 100, 2)
    top_spend = rng.randint(300000, 900000)
    
    metadata = {
        "entity": entity, "year": year, "supplier_id": supplier_id,
        "region_id": region_id, "market_id": market_id,
        "base_currency": base_currency, "current_rate": current_rate,
        "avg_rate": avg_rate, "pct_deviation": pct_deviation,
        "synthetic": True, "generator": "tool_r1.data_hard.v1"
    }
    template_index = _heldout_template_index(rng, heldout)
    docs = []
    source_ids = []
    
    # Company → Supplier
    cs_top = _company_supplier_doc(digest, "cs-top", entity, year, supplier_id, top_spend, split, is_top=True)
    docs.append(cs_top)
    source_ids.append(cs_top["id"])
    
    for index_d in range(2):
        other_supplier = f"SUP-{_hex_token(rng, 8)}"
        other_spend = rng.randint(100000, top_spend - 50000)
        docs.append(_company_supplier_doc(digest, f"cs-d{index_d}", entity, year,
                                          other_supplier, other_spend, split, is_top=False))
    
    # Supplier → Region
    sr = _supplier_region_doc(digest, "sr", supplier_id, region_id, split, entity)
    docs.append(sr)
    source_ids.append(sr["id"])
    
    for index_d in range(2):
        other_region = f"REG-{_hex_token(rng, 4)}"
        other_supplier = f"SUP-{_hex_token(rng, 8)}"
        docs.append(_supplier_region_doc(digest, f"sr-d{index_d}", other_supplier, other_region, split, entity))
    
    # Region → Market
    rm_doc = {
        "id": f"{digest}:rm", "type": "region_market", "split": split,
        "entity": entity, "synthetic": True,
        "record": {"region_id": region_id, "market_id": market_id},
        "text": f"Region {region_id} profile: serves primary market {market_id}.",
    }
    docs.append(rm_doc)
    source_ids.append(rm_doc["id"])
    
    for index_d in range(2):
        other_market = f"MKT-{_hex_token(rng, 4)}"
        other_region = f"REG-{_hex_token(rng, 4)}"
        docs.append({
            "id": f"{digest}:rm-d{index_d}", "type": "region_market", "split": split,
            "entity": entity, "synthetic": True,
            "record": {"region_id": other_region, "market_id": other_market},
            "text": f"Region {other_region} profile: serves primary market {other_market}.",
        })
    
    # Market → Current Exchange Rate
    ex_doc = {
        "id": f"{digest}:ex", "type": "exchange_rate", "split": split,
        "entity": entity, "synthetic": True,
        "record": {"market_id": market_id, "base_currency": base_currency, "rate": current_rate, "period": "current"},
        "text": f"Market {market_id} trading: {base_currency} exchange rate {current_rate:.6f} (current period).",
    }
    docs.append(ex_doc)
    source_ids.append(ex_doc["id"])
    
    for index_d in range(2):
        other_market = f"MKT-{_hex_token(rng, 4)}"
        other_rate = round(rng.uniform(0.5, 2.5), 6)
        docs.append({
            "id": f"{digest}:ex-d{index_d}", "type": "exchange_rate", "split": split,
            "entity": entity, "synthetic": True,
            "record": {"market_id": other_market, "base_currency": base_currency, "rate": other_rate},
            "text": f"Market {other_market} trading: {base_currency} exchange rate {other_rate:.6f}.",
        })
    
    # Historical average exchange rate
    hist_doc = {
        "id": f"{digest}:hist", "type": "historical_rate", "split": split,
        "entity": entity, "synthetic": True,
        "record": {"market_id": market_id, "base_currency": base_currency, "avg_rate": avg_rate, "years": 3},
        "text": f"Market {market_id} 3-year average {base_currency} rate: {avg_rate:.6f}.",
    }
    docs.append(hist_doc)
    source_ids.append(hist_doc["id"])
    
    target = _target(pct_deviation, "percent")
    sample = {
        "id": sample_id,
        "question": HARD_TEMPLATES["multi_hop_5"][template_index].format(**metadata),
        "task_type": "multi_hop_5",
        "split": split,
        "template_family": f"multi_hop_5_v{template_index + 1}",
        "template_status": "heldout" if heldout else "known",
        "source_doc_ids": source_ids,
        "answer": target,
        "metadata": metadata,
        "provenance": "synthetic; no real organizations or financial records"
    }
    
    actions = []
    if split == "train":
        actions = [
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"{entity} {year} top_supplier_id purchase", "top_k": 3}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"supplier {supplier_id} region_id profile", "top_k": 1}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"region {region_id} market_id profile", "top_k": 1}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"market {market_id} exchange_rate {base_currency}", "top_k": 1}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"market {market_id} 3-year average {base_currency} rate", "top_k": 1}},
            _final(target),
        ]
    
    demo = None
    if split == "train":
        demo = {
            "sample_id": sample_id,
            "split": "train",
            "origin": "synthetic deterministic teacher; replay before SFT",
            "reference_actions": actions
        }
    
    return sample, docs, demo


def _make_chain_of_tools(rng: random.Random, seed: int, split: str, index: int, digest: str,
                         heldout: bool) -> tuple[dict, list[dict], dict | None]:
    """Chain-of-tools task: retrieve → compute → conditional → final"""
    sample_id = f"{split}-{index:06d}"
    entity = f"NOVA-{digest}"
    year = rng.randrange(2010, 2045)
    supplier_id = f"SUP-{_hex_token(rng, 8)}"
    region_id = f"REG-{_hex_token(rng, 4)}"
    
    # Transaction data (retrieve)
    n_sales = rng.randint(3, 6)
    sale_boxes = [rng.randint(50, 200) for _ in range(n_sales)]
    n_returns = rng.randint(0, 2)
    return_boxes = [rng.randint(10, 50) for _ in range(n_returns)]
    total_boxes = sum(sale_boxes)  # No returns in tax calculation
    
    # Tax rate (retrieve from region)
    tax_rate = round(rng.uniform(0.05, 0.25), 1)
    
    # Compute tax liability
    tax_liability = round(total_boxes * tax_rate, 2)
    
    metadata = {
        "entity": entity, "year": year, "supplier_id": supplier_id,
        "region_id": region_id, "total_boxes": total_boxes,
        "tax_rate": tax_rate, "tax_liability": tax_liability,
        "synthetic": True, "generator": "tool_r1.data_hard.v1"
    }
    template_index = _heldout_template_index(rng, heldout)
    docs = []
    source_ids = []
    
    # Transaction document (sales and returns mixed)
    transactions = (
        [(f"SALE", rng.randint(50, 200)) for _ in range(n_sales)] +
        [(f"RETURN", rng.randint(10, 50)) for _ in range(n_returns)]
    )
    
    txn_doc = {
        "id": f"{digest}:txn", "type": "transaction_log", "split": split,
        "entity": entity, "synthetic": True,
        "record": {
            "entity": entity, "year": year, "transactions": transactions
        },
        "text": f"Transaction log for {entity} {year}: " + "; ".join(
            f"{typ} {amt} boxes" for typ, amt in transactions
        ) + ".",
    }
    docs.append(txn_doc)
    source_ids.append(txn_doc["id"])
    
    # Distractor transaction
    other_entity = f"NOVA-{_hex_token(rng, 16)}"
    docs.append({
        "id": f"{digest}:txn-d", "type": "transaction_log", "split": split,
        "entity": other_entity, "synthetic": True,
        "record": {"entity": other_entity, "year": year},
        "text": f"Transaction log for {other_entity} {year}: unrelated.",
    })
    
    # Tax rate document
    tax_doc = {
        "id": f"{digest}:tax", "type": "tax_rate", "split": split,
        "entity": entity, "synthetic": True,
        "record": {"region_id": region_id, "rate": tax_rate},
        "text": f"Region {region_id} tax rate: {tax_rate}% on box sales.",
    }
    docs.append(tax_doc)
    source_ids.append(tax_doc["id"])
    
    # Distractor tax rates
    for index_d in range(2):
        other_region = f"REG-{_hex_token(rng, 4)}"
        other_rate = round(rng.uniform(0.05, 0.25), 1)
        docs.append({
            "id": f"{digest}:tax-d{index_d}", "type": "tax_rate", "split": split,
            "entity": entity, "synthetic": True,
            "record": {"region_id": other_region, "rate": other_rate},
            "text": f"Region {other_region} tax rate: {other_rate}% on box sales.",
        })
    
    target = _target(tax_liability, "credits")
    sample = {
        "id": sample_id,
        "question": HARD_TEMPLATES["chain_of_tools"][template_index].format(**metadata),
        "task_type": "chain_of_tools",
        "split": split,
        "template_family": f"chain_of_tools_v{template_index + 1}",
        "template_status": "heldout" if heldout else "known",
        "source_doc_ids": source_ids,
        "answer": target,
        "metadata": metadata,
        "provenance": "synthetic; no real organizations or financial records"
    }
    
    actions = []
    if split == "train":
        # Step 1: retrieve transaction log
        # Step 2: compute total (exclude returns) - would be done by Python tool in real setting
        # Step 3: retrieve tax rate for region
        # Step 4: compute liability = total * rate
        python_code = (
            f"transactions = {transactions!r}\n"
            f"total_sales = sum(amt for typ, amt in transactions if typ == 'SALE')\n"
            f"tax_rate = {tax_rate}\n"
            f"liability = round(total_sales * tax_rate, 2)\n"
            f"print(liability)\n"
        )
        actions = [
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"{entity} {year} transaction log sales", "top_k": 1}},
            {"action": "tool", "name": "retrieve", "arguments": {"query": f"region {region_id} tax_rate {year}", "top_k": 1}},
            {"action": "tool", "name": "python", "arguments": {"code": python_code}},
            _final(target),
        ]
    
    demo = None
    if split == "train":
        demo = {
            "sample_id": sample_id,
            "split": "train",
            "origin": "synthetic deterministic teacher; replay before SFT",
            "reference_actions": actions
        }
    
    return sample, docs, demo


def generate_hard_dataset(output_dir: str | Path, train_size: int = 1200, seed: int = 42, version: str = "train-hard-v2") -> dict:
    """Generate hard difficulty training dataset with new complex task types.
    
    This is designed to create gradient signal for GRPO by:
    1. Requiring multiple sequential tool calls (harder to answer randomly)
    2. Including distractors that waste tokens if retrieved
    3. Shifting reward to emphasize cost optimization
    
    Only generates train split; validation/test remain unchanged from original.
    This dataset bypasses normal audit rules and requires SKIP_DATA_AUDIT=1 for training.
    
    Args:
        output_dir: Output directory for the dataset
        train_size: Number of training samples to generate
        seed: Random seed for reproducibility
        version: Dataset version - "train-hard-v2" (original) or "train-hard-v3" (expanded multi_hop_4)
    """
    from .data import _make_sample as make_sample_original
    
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    # Only generate training split
    split = "train"
    directory = root / split
    directory.mkdir(exist_ok=True)
    
    rng = random.Random(f"tool-r1:hard:{seed}:{split}")
    
    # Task distribution: use specified version or default to v2
    plan = HARD_TASK_PLAN.get(version, HARD_TASK_PLAN["train-hard-v2"])
    counts = _plan_split_counts(train_size, plan)
    
    task_list = []
    for task, amount in counts.items():
        task_list.extend([task] * amount)
    rng.shuffle(task_list)
    
    samples, corpus, demonstrations = [], [], []
    
    for index in range(train_size):
        task = task_list[index] if index < len(task_list) else "direct"
        digest = hashlib.sha256(f"{seed}:hard:{split}:{index}".encode()).hexdigest()[:16]
        heldout = False  # Hard dataset is train-only, no heldout
        
        # Dispatch to task-specific generator
        if task == "multi_hop_4":
            sample, docs, demo = _make_multi_hop_4(rng, seed, split, index, digest, heldout)
        elif task == "multi_hop_5":
            sample, docs, demo = _make_multi_hop_5(rng, seed, split, index, digest, heldout)
        elif task == "chain_of_tools":
            sample, docs, demo = _make_chain_of_tools(rng, seed, split, index, digest, heldout)
        else:
            # Use original data generator for standard tasks
            sample, docs, demo = make_sample_original(rng, seed, split, index, task)
        
        samples.append(sample)
        corpus.extend(docs)
        if demo is not None:
            demonstrations.append(demo)
    
    # Shuffle before writing
    rng.shuffle(samples)
    rng.shuffle(corpus)
    
    # Write to files
    paths = [
        (directory / "samples.jsonl", samples),
        (directory / "corpus.jsonl", corpus),
        (directory / "demonstrations.jsonl", demonstrations),
    ]
    
    files = {}
    for path, rows in paths:
        _write_rows(path, rows)
        files[path.relative_to(root).as_posix()] = {"sha256": _sha256(path), "rows": len(rows)}
    
    # Create manifest
    task_counts = {}
    for sample in samples:
        task_type = sample["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    split_summary = {
        split: {
            "samples": len(samples),
            "documents": len(corpus),
            "tasks": task_counts,
            "templates": {"known": len([s for s in samples if s["template_status"] == "known"])},
        }
    }
    
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "dataset_type": "hard",
        "source": "Synthetic hard tasks for GRPO training with increased tool complexity",
        "task_plan": HARD_TASK_PLAN,
        "files": files,
        "splits": split_summary,
        "skip_normal_audit": True,  # Hard datasets use a relaxed audit path
    }
    
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    
    _dump(root / "manifest.json", manifest)
    
    return manifest
