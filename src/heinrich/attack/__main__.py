"""Run attack analysis as a module.

Usage:
    python -m heinrich.attack --model X --db path
"""
from __future__ import annotations

import argparse
import json

from heinrich.attack.run import attack_to_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack analysis on a model")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--db", default=None, help="Database path")
    args = parser.parse_args()

    result = attack_to_db(args.model, db_path=args.db)
    print(json.dumps(result, indent=2))
