import os
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # pinned; the benchmarked weights
MODEL_SIZE = "~9 GB"

HOST = "127.0.0.1"
PORT = int(os.environ.get("DECIDE_PORT", "8792"))
URL = f"http://{HOST}:{PORT}"

STATE_DIR = Path(os.environ.get("DECIDE_HOME", Path.home() / ".decide"))
PID_FILE = STATE_DIR / "server.pid"
LOG_FILE = STATE_DIR / "server.log"
