"""Compose every launcher option against the installed pinned verl config tree.

This check does not load a model or allocate GPU memory. --config-dir exists
for an offline upstream-config fixture and must point to verl/trainer/config.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from tool_r1.training.launch import ROOT, build_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir")
    args = parser.parse_args()
    if args.config_dir:
        upstream = Path(args.config_dir).resolve()
    else:
        spec = importlib.util.find_spec("verl")
        if spec is None:
            raise RuntimeError("Install the pinned server stack first")
        upstream = Path(spec.origin).parent / "trainer/config"
    config = json.loads((ROOT / "configs/a6000.json").read_text())
    with initialize_config_dir(version_base=None, config_dir=str(upstream)):
        for mode in ("M3", "M4", "M5"):
            overrides, _ = build_overrides(config, model=ROOT / "outputs/sft/merged", data=ROOT / "data",
                prepared=ROOT / "data/prepared", run_dir=ROOT / "outputs/config-check", mode=mode)
            composed = compose(config_name="ppo_trainer", overrides=overrides)
            assert composed.algorithm.adv_estimator == "grpo"
            assert composed.actor_rollout_ref.rollout.n == 4
            assert composed.actor_rollout_ref.model.override_config.attn_implementation == "sdpa"
            print(f"{mode}: composed {len(overrides)} overrides successfully")


if __name__ == "__main__":
    main()
