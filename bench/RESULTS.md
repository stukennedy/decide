# Scorer experiments (26 Sep 2026)

`decide` scores a decision by reading the model's next-token logits for the option letters
(the method SemIf uses). We reimplemented it on MLX-LM and tried to improve it.

**Protocol.** Settings were compared on a dev set that shares nothing with JevBench, then only
the best candidates were run on JevBench's 231 public tasks, the final test. All runs use
`Qwen/Qwen3.5-4B` at revision `851bf6e`, BF16, MLX, on an M5 Max.

- Dev set (508 labelled, all 3-option): SemIf's `authored144` and `perturbations108`
  fixtures, and 256 WANLI test items selected by SemIf's manifest
  (WANLI test.jsonl sha256 `4276e0af…`, fetched from the pinned Hugging Face revision).
- JevBench: commit `f8ce713`, graded by its own `semif_direct` adapter with our scorer
  swapped in (`bench/jevbench_run.py`).

## Results

| Scorer | Dev acc | Dev NLL | JevBench | Original | Easy | Hard | p50 |
|---|---|---|---|---|---|---|---|
| SemIf (reference) | 70.7% | 0.787 | 187/231 | 71/72 | 48/48 | 68/111 | 0.06 s |
| **json, 1 order (default)** | 70.7% | 0.787 | **186/231** | 71/72 | 48/48 | 67/111 | 0.08 s |
| json, 2 orders | 74.6% | 0.654 | 183/231 | 69/72 | 48/48 | 66/111 | 0.16 s |
| text, 1 order | **78.3%** | 0.713 | 176/231 | 60/72 | 48/48 | 68/111 | 0.09 s |
| text, 2 orders | 77.8% | 0.663 | 174/231 | 61/72 | 48/48 | 65/111 | 0.15 s |
| text, question first | 73.8% | 0.713 | not run | | | | |
| text, no system message | 75.0% | 0.765 | not run | | | | |

`json, 1 order` matches SemIf exactly on the dev set (same prediction on 508/508, probabilities
equal to 4 d.p.) and on 230/231 JevBench tasks.

## Findings

- **Dev-set gains didn't transfer.** The plain-text prompt was 7.6 points better on the dev set
  and 10 tasks worse on JevBench, almost entirely in the original tier (adequacy, policy, intent:
  yes/no and routing questions the 3-option dev set doesn't contain). A dev set that doesn't look
  like the target traffic isn't a reliable guide for prompt wording on a 4B model.
- **Averaging over option orders** reduces letter bias and improves probability quality on the dev set
  (NLL 0.787 → 0.654) but didn't improve JevBench accuracy, at twice the cost. It's available as
  `DECIDE_ORDERS=2` for people who care more about well-behaved probabilities than speed.
- The prompt wording moves accuracy by several points in either direction. Any future change should be
  checked on JevBench-like data (yes/no, routing, many-option choice) before it ships.

## Reproduce

```sh
# dev set
PYTHONPATH=~/decide python bench/dev_eval.py authored144.jsonl perturbations108.jsonl wanli256.jsonl \
  --configs json:1 json:2 text:1 text:2
# JevBench (from the jevbench checkout)
DECIDE_PROMPT=json DECIDE_ORDERS=1 PYTHONPATH=.:~/decide python ~/decide/bench/jevbench_run.py run \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --adapter semif_direct --endpoint Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --key-env '' --reserve-usd 0 --results runs/decide/results.jsonl
```
`jevbench_run.py` needs `semif_phase1` importable, because JevBench's adapter imports it; it's only
used as the hook point, not for scoring.
