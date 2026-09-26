import os
import subprocess
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


def _ram_gb():
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    except (OSError, ValueError):
        return 0


def model_bits():
    """Weight precision: DECIDE_BITS=16|8|4, else 8-bit on Macs with 24 GB of RAM or less.

    Full precision needs about 8 GB of memory, 8-bit about 4.5 GB and 4-bit about 2.5 GB.
    On JevBench's 231 public tasks: 186, 185 and 180 correct.
    """
    value = os.environ.get("DECIDE_BITS", "").strip().lower()
    if value in ("16", "bf16", "full"):
        return None
    if value in ("8", "4"):
        return int(value)
    if value:
        raise ValueError("DECIDE_BITS must be 16, 8 or 4")
    return 8 if 0 < _ram_gb() <= 24 else None
