"""Run JevBench with decide's scorer, graded by JevBench's own semif_direct adapter.

The adapter maps each task to options and calls semif_phase1's loader and scorer at run
time; we point those names at decide.scorer so everything else (task mapping, grading,
result files) is exactly what produced the earlier SemIf numbers.

usage (from the jevbench checkout, with PYTHONPATH including it and ~/decide):
  DECIDE_PROMPT=text DECIDE_ORDERS=1 python ~/decide/bench/jevbench_run.py run --tasks ... \
      --adapter semif_direct --endpoint Qwen/Qwen3.5-4B --revision <rev> --results ...
"""
import os
import sys

import semif_phase1.core as core
import semif_phase1.direct as direct

from decide.scorer import Scorer

PROMPT = os.environ.get("DECIDE_PROMPT", "text")
ORDERS = int(os.environ.get("DECIDE_ORDERS", "1"))
BITS = int(os.environ["DECIDE_BITS"]) if os.environ.get("DECIDE_BITS") else None


def load(source, revision, *args, **kwargs):
    scorer = Scorer.load(source, revision, prompt=PROMPT, orders=ORDERS, bits=BITS)
    return scorer, None, {"source": source, "revision": revision, "backend": "mlx", "scorer": "decide",
                          "prompt": PROMPT, "orders": ORDERS, "bits": BITS}


def score(scorer, _tokenizer, row, meta, max_tokens=4096):
    scorer.max_tokens = max_tokens
    out = scorer.score(row)
    out.update(option_logits=None, forward_seconds=out["total_seconds"], prompt_sha256=None,
               prompt_version=f"decide-{PROMPT}-o{ORDERS}", model=meta,
               readout="letter logits, averaged over option orders")
    return out


core.load_causal_model = load
direct.score = score

from jevbench.cli import main  # noqa: E402

sys.exit(main())
