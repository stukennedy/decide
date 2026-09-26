"""Compare scorer settings on a labelled dev set (never JevBench, which is kept for the final test).

usage: python bench/dev_eval.py DEV.jsonl [DEV2.jsonl ...] --configs text:1 text:2 json:1 ...
Each row needs state, question, options and an integer `label` (index of the right option).
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from decide.config import MODEL, REVISION
from decide.scorer import Scorer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--configs", nargs="+", default=["json:1", "text:1", "text:2"])
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    rows = []
    for f in args.files:
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                r["_set"] = f.stem
                rows.append(r)
    base = Scorer.load(MODEL, REVISION)
    report = {}
    for cfg in args.configs:
        prompt, orders = cfg.split(":")
        scorer = Scorer(base.model, base.tokenizer, prompt=prompt, orders=int(orders))
        scorer.score(rows[0])  # warm up kernels for this shape
        per_set, times, preds = {}, [], []
        for r in rows:
            t = time.perf_counter()
            res = scorer.score(r)
            times.append(time.perf_counter() - t)
            probs = res["probabilities"]
            pred = max(range(len(probs)), key=probs.__getitem__)
            s = per_set.setdefault(r["_set"], {"n": 0, "correct": 0, "nll": 0.0})
            s["n"] += 1
            s["correct"] += pred == r["label"]
            s["nll"] += -math.log(max(probs[r["label"]], 1e-12))
            preds.append({"id": r["id"], "set": r["_set"], "pred": pred, "label": r["label"], "probs": probs})
        n = len(rows)
        correct = sum(s["correct"] for s in per_set.values())
        nll = sum(s["nll"] for s in per_set.values()) / n
        line = f"{cfg:<10} acc {correct}/{n} ({100 * correct / n:.1f}%)  nll {nll:.3f}  p50 {statistics.median(times) * 1000:.0f} ms  |  " + \
               "  ".join(f"{k} {v['correct']}/{v['n']}" for k, v in per_set.items())
        print(line, flush=True)
        report[cfg] = {"correct": correct, "n": n, "nll": nll, "p50_s": statistics.median(times),
                       "sets": per_set, "predictions": preds}
    if args.out:
        args.out.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
