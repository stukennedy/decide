#!/usr/bin/env python3
"""Claude Code PreToolUse hook: pick a cheaper model for each new subagent with decide.

Registered on the `Agent` tool. Before a subagent starts, it asks the local decide server
two questions about the task and sets the subagent's model:

  hard (design, subtle debugging, security review)           -> opus
  polished text for people (docs, emails, copy, release notes) -> sonnet
  otherwise: simple -> haiku, standard -> sonnet

It never blocks: if decide isn't running, isn't confident, or anything goes wrong, the
subagent starts exactly as Claude asked. A model Claude chose explicitly is kept unless
DECIDE_ROUTER_OVERRIDE=1. Decisions are logged to ~/.decide/router.log.

Settings (environment variables):
  DECIDE_URL               default http://127.0.0.1:8792
  DECIDE_ROUTER_MIN_CONF   minimum probability to act on a decision, default 0.6
  DECIDE_ROUTER_OVERRIDE   1 to replace a model Claude chose explicitly
  DECIDE_ROUTER_DRY_RUN    1 to log decisions without changing anything
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("DECIDE_URL", "http://127.0.0.1:8792")
MIN_CONF = float(os.environ.get("DECIDE_ROUTER_MIN_CONF", "0.6"))
OVERRIDE = os.environ.get("DECIDE_ROUTER_OVERRIDE") == "1"
DRY_RUN = os.environ.get("DECIDE_ROUTER_DRY_RUN") == "1"
LOG = Path(os.environ.get("DECIDE_HOME", Path.home() / ".decide")) / "router.log"
MAX_CHARS = 6000  # keeps the request well inside decide's 4,096-token limit
# decide runs on this machine: never send its requests through HTTP_PROXY or a system proxy
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))

WRITING = {
    "question": "Will the result of this task be polished text that is published or sent to people, "
                "such as documentation, an email or marketing copy?",
    "options": [
        {"id": "writing", "description": "Yes: documentation, a README, an email, marketing or landing "
                                         "page copy, release notes, a blog post or a commit message"},
        {"id": "other", "description": "No: code changes, commands, searching, investigation, or findings "
                                       "and notes reported back to the assistant"},
    ],
}
DIFFICULTY = {
    "question": "How much skill and reasoning does this task need from the assistant doing it?",
    "options": [
        {"id": "simple", "description": "Simple: look something up, find files, read and summarise, "
                                        "a small mechanical edit, or run a command and report back"},
        {"id": "standard", "description": "Standard: normal development work such as implementing a "
                                          "feature, writing tests or fixing an ordinary bug"},
        {"id": "hard", "description": "Hard: subtle debugging, architecture or design decisions, "
                                      "security review, concurrency, or reasoning across many files"},
    ],
}
MODEL_FOR = {"writing": "sonnet", "simple": "haiku", "standard": "sonnet", "hard": "opus"}


def post(path, body, timeout):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with LOCAL.open(req, timeout=timeout) as r:
        return json.load(r)


def server_up():
    try:
        with LOCAL.open(URL + "/health", timeout=2) as r:
            return json.load(r).get("status") == "ready"
    except OSError:
        return False


def ensure_server():
    if server_up():
        return True
    exe = shutil.which("decide") or str(Path.home() / ".local/bin/decide")
    if not Path(exe).exists():
        return False
    subprocess.run([exe, "start"], capture_output=True, timeout=120)
    return server_up()


def ask(task, spec):
    r = post("/score", {"id": "route", "state": task, **spec}, timeout=10)
    best = max(zip(r["option_ids"], r["probabilities"]), key=lambda x: x[1])
    return best, dict(zip(r["option_ids"], (round(p, 3) for p in r["probabilities"])))


def log(entry):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def route(task):
    """Return (model or None, reason, details). None means leave the subagent as it is."""
    # hard reasoning wins even when it ends in a write-up (design docs, security reviews)
    (level, p_level), levels = ask(task, DIFFICULTY)
    details = {"difficulty": levels}
    if level == "hard" and p_level >= MIN_CONF:
        return MODEL_FOR["hard"], "hard", details
    (kind, p_kind), details["writing"] = ask(task, WRITING)
    if kind == "writing" and p_kind >= MIN_CONF:
        return MODEL_FOR["writing"], "writing", details
    if p_level >= MIN_CONF:
        return MODEL_FOR[level], level, details
    return None, "not confident", details


def main():
    event = json.load(sys.stdin)
    if event.get("tool_name") != "Agent":
        return
    tool_input = event.get("tool_input") or {}
    entry = {"ts": time.time(), "description": tool_input.get("description", "")}
    if tool_input.get("model") and not OVERRIDE:
        log({**entry, "kept": tool_input["model"], "reason": "model chosen by Claude"})
        return
    task = f"{tool_input.get('description', '')}\n\n{tool_input.get('prompt', '')}".strip()[:MAX_CHARS]
    if not task or not ensure_server():
        return
    model, reason, details = route(task)
    log({**entry, "model": model, "reason": reason, **details, "dry_run": DRY_RUN})
    if model is None or DRY_RUN:
        return
    # no permissionDecision: the normal permission rules still apply to the modified call
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "updatedInput": {**tool_input, "model": model},
    }}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never break an agent launch
        log({"ts": time.time(), "error": f"{type(e).__name__}: {e}"})
    sys.exit(0)
