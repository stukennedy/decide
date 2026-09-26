# Route Claude Code subagents to cheaper models

`route_subagent.py` is a Claude Code `PreToolUse` hook. Each time Claude starts a subagent,
it asks your local `decide` server what kind of task it is (about 0.1 s, free) and sets the
subagent's model:

| The task is | Model |
|---|---|
| Hard: subtle debugging, design, security review, reasoning across many files | `opus` |
| Polished text for people: docs, READMEs, emails, copy, release notes, commit messages | `sonnet` |
| Standard development: features, tests, ordinary bugs | `sonnet` |
| Simple: find files, look things up, summarise, small edits, run a command | `haiku` |

It never blocks an agent. If `decide` isn't running (it tries `decide start` first), isn't
confident (below 0.6), or anything fails, the subagent starts exactly as Claude asked. If
Claude picks a model explicitly, that choice is kept.

## Install

Needs `decide` installed (`curl -fsSL https://decide.run/install | sh`). Add this to
`~/.claude/settings.json` for every project, or `.claude/settings.json` for one:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Agent",
        "hooks": [
          { "type": "command", "command": "python3 /path/to/route_subagent.py", "timeout": 150 }
        ]
      }
    ]
  }
}
```

Every decision is logged to `~/.decide/router.log`:

```json
{"description": "Generate commit message for config flag rename", "model": "sonnet", "reason": "writing",
 "difficulty": {"simple": 0.64, "standard": 0.343, "hard": 0.017}, "writing": {"writing": 0.986, "other": 0.014}}
```

## Settings

| Environment variable | Effect |
|---|---|
| `DECIDE_ROUTER_DRY_RUN=1` | Log decisions without changing anything, to see what it would do first |
| `DECIDE_ROUTER_OVERRIDE=1` | Also replace models Claude chose explicitly |
| `DECIDE_ROUTER_MIN_CONF=0.6` | How sure decide must be before acting |

## How well it works

- **Routing:** 23/24 on hand-labelled tasks used while writing the questions (the miss sends
  "summarise a file" to Sonnet instead of Haiku), and **12/12 on fresh tasks** written afterwards
  and run once. Run `python3 check_routing.py` or pass your own JSONL of
  `{"expect", "description", "prompt"}` rows.
- **End to end:** with Claude Code 2.1.282, a session on Haiku delegating a commit message got
  a subagent on `claude-sonnet-5` (read from the subagent's transcript). The same run with
  `DECIDE_ROUTER_DRY_RUN=1` got a Haiku subagent, so the hook is what changed it.

## Caveats

- Setting `model` through `updatedInput` works but isn't a documented contract for the `Agent`
  tool; a Claude Code update could change it. Dry-run mode and the log make that easy to spot.
- The chosen model takes priority over a model set in a custom subagent's frontmatter.
- A small model judges the task from its description and prompt, so vague delegation
  ("look into this") is routed on thin evidence. The confidence threshold covers the worst cases.
