from __future__ import annotations

import argparse
import json

import numpy as np

from .config import load_config, validate_config
from .mcc import greedy_mcc
from .statistics import mmd_permutation_test, paired_sign_flip_test


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM integrity fingerprint toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    subparsers.add_parser("stats-demo")
    subparsers.add_parser("mcc-demo")
    args = parser.parse_args()
    if args.command == "validate-config":
        config = load_config(args.config)
        validate_config(config)
        print(json.dumps({"valid": True, "config": config["_config_path"]}, ensure_ascii=False))
    elif args.command == "stats-demo":
        rng = np.random.default_rng(42)
        x = rng.normal(size=(16, 8))
        y = x + 0.75
        print(json.dumps(mmd_permutation_test(x, y, 199).as_dict(), indent=2))
        print(json.dumps(paired_sign_flip_test(x, y, 199).as_dict(), indent=2))
    elif args.command == "mcc-demo":
        value = greedy_mcc({"a": {"attention:0:0"}, "b": {"ffn:0:1", "ffn:0:2"}}, 1)
        print(json.dumps({"selected": value.selected_ids, "gains": value.marginal_gains}, indent=2))


if __name__ == "__main__":
    main()
