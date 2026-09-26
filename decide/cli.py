"""decide: fast local decisions from the command line.

Standard library only at import time, so each command starts quickly. The model lives in a
background server (`decide start`) that the other commands start automatically.
"""
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import __version__
from .config import LOG_FILE, MODEL, MODEL_SIZE, PID_FILE, REVISION, STATE_DIR, URL
from .rows import ask_row, ranked

EPILOG = """examples:
  decide ask "Customer: I was charged twice for order 1182." "Which team?" billing shipping tech
  decide ask -q "$(cat ticket.txt)" "Is this urgent?" urgent not_urgent
  cat email.txt | decide ask - "Does it ask for a refund?" "yes=asks for a refund" "no=does not"
  decide score decisions.jsonl          (one {"state","question","options"} object per line)
  decide status | decide stop
"""


class DecideError(Exception):
    pass


# ---------------------------------------------------------------- server lifecycle

def _request(path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            message = json.load(e).get("error", str(e))
        except ValueError:
            message = str(e)
        raise DecideError(message) from None


def health():
    try:
        return _request("/health", timeout=2)
    except (OSError, DecideError):
        return None


def _pid():
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _model_cached():
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any miss means "download needed"
        return False


def download_model():
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL} ({MODEL_SIZE}, first run only)...", file=sys.stderr)
    snapshot_download(MODEL, revision=REVISION)


def check_platform():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise DecideError("decide needs an Apple Silicon Mac (it runs the model on MLX)")


def start(quiet=False):
    if health():
        return
    check_platform()
    if not _model_cached():
        download_model()
    proc = None
    if not _pid():  # otherwise a server is already loading; just wait for it
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "ab") as log:
            proc = subprocess.Popen([sys.executable, "-m", "decide.server"], stdout=log, stderr=log,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
    if not quiet:
        print("Starting decide server (loading the model takes a few seconds)...", file=sys.stderr)
    deadline = time.time() + 180
    while time.time() < deadline:
        if health():
            return
        if proc and proc.poll() is not None:
            raise DecideError(f"server exited during startup; see {LOG_FILE}")
        time.sleep(0.3)
    raise DecideError(f"server didn't start within 3 minutes; see {LOG_FILE}")


def stop():
    pid = _pid()
    if not pid:
        print("decide server is not running")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _pid():
            break
        time.sleep(0.1)
    print("decide server stopped")


# ---------------------------------------------------------------- commands

def cmd_ask(args):
    state = sys.stdin.read() if args.state == "-" else args.state
    row = ask_row(state, args.question, args.options)
    start()
    result = _request("/score", row)
    if args.json:
        print(json.dumps(result))
        return
    best = ranked(result)
    if args.quiet:
        print(best[0][0])
        return
    print(best[0][0])
    width = max(len(k) for k, _ in best)
    for k, p in best:
        print(f"  {k:<{width}}  {p:.3f}")


def _read_rows(path):
    stream = sys.stdin if path == "-" else open(path)
    with stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("id", str(i + 1))
    return rows


def cmd_score(args):
    rows = _read_rows(args.file)
    if not rows:
        raise DecideError("no rows to score")
    start()
    results = _request("/score", rows)
    for r in results:
        if args.json:
            print(json.dumps(r))
        else:
            best = ranked(r)
            print(f"{r['id']}\t{best[0][0]}\t" + "  ".join(f"{k} {p:.3f}" for k, p in best))


def cmd_status(args):
    h = health()
    if h:
        print(f"running at {URL} (pid {h.get('pid')}) - {h['model']} on {h['backend']}, decide {h.get('version')}")
    else:
        print("not running (starts automatically on the next ask/score, or run `decide start`)")
        sys.exit(1)


def cmd_start(args):
    start()
    print(f"decide server ready at {URL}")


def cmd_setup(args):
    check_platform()
    if not _model_cached():
        download_model()
    start()
    result = _request("/score", ask_row("The deploy finished and every health check passed.",
                                        "Did the deploy succeed?", ["yes", "no"]))
    print(f"decide is ready: test decision -> {ranked(result)[0][0]} "
          f"({result['total_seconds'] * 1000:.0f} ms). Try: decide ask --help")


def main(argv=None):
    p = argparse.ArgumentParser(prog="decide", description="Fast local decisions: pick one of your options, "
                                "with probabilities. Runs Qwen3.5-4B on your Mac with MLX.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"decide {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="decide one question", epilog=EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("state", help="the context to decide about; '-' reads it from stdin")
    a.add_argument("question", help="the question, e.g. 'Which team should handle this?'")
    a.add_argument("options", nargs="+", metavar="option",
                   help="two or more options: 'id' or 'id=description'")
    out = a.add_mutually_exclusive_group()
    out.add_argument("-q", "--quiet", action="store_true", help="print only the winning option")
    out.add_argument("--json", action="store_true", help="print the full result as JSON")
    a.set_defaults(func=cmd_ask)

    s = sub.add_parser("score", help="decide many questions from a JSONL file")
    s.add_argument("file", help="JSONL file, one {id?, state, question, options} per line; '-' for stdin")
    s.add_argument("--json", action="store_true", help="print full results as JSONL")
    s.set_defaults(func=cmd_score)

    sub.add_parser("setup", help="download the model and run a test decision").set_defaults(func=cmd_setup)
    sub.add_parser("start", help="start the background server").set_defaults(func=cmd_start)
    sub.add_parser("stop", help="stop the background server and free its memory").set_defaults(
        func=lambda _: stop())
    sub.add_parser("status", help="show whether the server is running").set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (DecideError, ValueError) as e:
        sys.exit(f"decide: {e}")
    except OSError as e:
        sys.exit(f"decide: {e.strerror or e}: {e.filename}" if e.filename else f"decide: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
