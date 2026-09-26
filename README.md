<img src="docs/icon-512.png" width="88" alt="decide logo: a d followed by a green block cursor">

# decide

Fast local decisions from the command line. Give it some context, a question and your options;
it picks one and tells you how sure it is. It runs on your Mac: no API key, nothing leaves the machine.

```console
$ decide ask "Customer: I was charged twice for order 1182." "Which team?" billing shipping tech
billing
  billing   1.000
  shipping  0.000
  tech      0.000
```

About 0.08 s per decision once the model is loaded. Under the hood it runs `Qwen/Qwen3.5-4B` with
Apple's MLX and reads the model's preference for each option directly: one forward pass, no text generated.

## Install

Needs an Apple Silicon Mac (M1 or later) and about 9 GB of disk. It uses about 8 GB of memory, or about
4.5 GB on Macs with 24 GB of RAM or less, where it loads the model at 8-bit precision (see below).

```sh
curl -fsSL https://decide.run/install | sh
```

The script installs [uv](https://docs.astral.sh/uv/) if you don't have it, installs `decide` into its own
Python 3.12 environment, downloads the pinned model (~9 GB, first time only) and runs a test decision.
If you'd rather do it by hand:

```sh
uv tool install --python 3.12 git+https://github.com/stukennedy/decide
decide setup
```

## Use

```sh
# options are 'id' or 'id=description'
decide ask "Policy: refunds need a receipt. Customer has no receipt." "Is a refund allowed?" \
  "allowed=refund is permitted" "denied=refund is not permitted"

# -q prints only the winner, handy in scripts
team=$(decide ask -q "$(cat ticket.txt)" "Which team?" billing shipping tech)

# '-' reads the context from stdin
cat email.txt | decide ask - "Does it ask for a refund?" yes no

# many decisions in one go: one {"id", "state", "question", "options"} object per line
decide score decisions.jsonl
decide score decisions.jsonl --json      # full results as JSONL
```

A `decisions.jsonl` row looks like:

```json
{"id": "route-1", "state": "Customer can't log in after a password reset.", "question": "Which queue?",
 "options": [{"id": "access", "description": "Account access"}, {"id": "billing", "description": "Billing"}]}
```

### The background server

The model stays loaded in a small background server, so each command answers in well under a second.
It starts automatically the first time you run `ask` or `score` (about 5 s) and keeps running until you stop it.

```sh
decide status     # is it running?
decide stop       # free the memory
decide start      # start it ahead of time
```

It listens on `127.0.0.1:8792` only (set `DECIDE_PORT` to change it) and has no authentication, so don't
expose the port. Logs are in `~/.decide/server.log`. Apps can call it directly:

- `POST /score` takes a row (or a list of rows) in the format above.
- `POST /v1/systemone` takes Jev-style requests (`{"state", "questions"}` with `noul`, `choice` and `score`
  questions) and returns Jev-style answers, so code written for Jev can point at it.

## Claude Code: route subagents to cheaper models

[`integrations/claude-code`](integrations/claude-code) has a hook that asks `decide` what kind of
task each new Claude Code subagent has, and sends it to Haiku, Sonnet or Opus to match: simple
lookups to Haiku, writing and everyday coding to Sonnet, hard reasoning to Opus.

## How good is it?

Measured on the 231 public [JevBench](https://github.com/fstandhartinger/jevbench) tasks (commit `f8ce713`),
on an M5 Max:

| | All | Original | Easy | Hard | p50 |
|---|---|---|---|---|---|
| **decide** (Qwen3.5-4B, MLX) | 186/231 | 71/72 | 48/48 | 67/111 | 0.08 s |
| Jev 1.13 (hosted API) | 198/231 | 71/72 | 48/48 | 79/111 | 0.62 s |
| Open-Jev 27B (Mac, MPS) | 198/231 | 69/72 | 48/48 | 81/111 | 1.36 s |

It's as good as the big models on short, well-specified decisions: routing, policy checks, intent,
extraction, tool selection. It's noticeably weaker on long documents and multi-step reasoning (the hard tier).
Use a bigger model for those.

Things to know:

- **Probabilities aren't calibrated.** The chosen option is reliable; don't read 0.95 as "95% chance of being
  right" until you've checked it on your own labelled examples.
- **Option wording and order matter** for a small model. Give options clear names or descriptions, and keep
  their order consistent.
- **Inputs are limited to 4,096 tokens** (roughly 3,000 words). Longer inputs are rejected, not truncated.
  `DECIDE_MAX_TOKENS` raises the limit, at the cost of speed and accuracy.

## Memory and precision

| Precision | Memory | JevBench (231 tasks) | Used by default on |
|---|---|---|---|
| Full (BF16) | ~8 GB | 186 (short 119/120, hard 67/111) | Macs with more than 24 GB of RAM |
| 8-bit | ~4.5 GB | 185 (short 118/120, hard 67/111) | Macs with 24 GB or less |
| 4-bit | ~2.5 GB | 180 (short 117/120, hard 63/111) | only if you ask for it |

Choose for yourself with `decide start --bits 16`, `8` or `4` (run `decide stop` first if a server
is already running), or set `DECIDE_BITS` so servers started automatically use it too.
`decide status` shows which one is running.

The benchmark times on this page were measured on an M5 Max. Smaller chips are slower: a short
decision takes about 0.04 s of model time on an M5 Max (full precision) and about 0.35 s on a base
M4 Mac mini (16 GB, 8-bit). Add about 0.1 s for the command itself, and expect the first decision
after starting to take about twice as long. Accuracy is the same on any Mac.
If answers are slow or `decide` seems stuck, the Mac is probably short of memory: check
`memory_pressure` and quit something large, like a VM or Docker.

## Uninstall

```sh
decide stop
uv tool uninstall decide-cli
rm -rf ~/.decide
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B    # the model, ~9 GB
```

## Credits and licence

`decide` is MIT licensed. It stands on other people's work:

- The scoring method (lettered options, reading the answer-letter logits) follows
  [SemIf](https://github.com/TheoLeeCJ/openjev) by Theodore Lee (MIT). `decide` has its own implementation,
  which matches SemIf's answers on 230 of 231 JevBench tasks. See [`bench/RESULTS.md`](bench/RESULTS.md) for
  the attempts to improve on it.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache-2.0) is the model, pinned to
  revision `851bf6e`.
- [MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm) by Apple (MIT) run it.

Not affiliated with TypeSafe AI or Jev.
