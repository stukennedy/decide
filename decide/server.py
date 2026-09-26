"""Background server that keeps the model loaded. Started by `decide start`.

  GET  /health          -> {"status": "ready", ...}
  POST /score           -> a row {"id","state","question","options":[{"id","description"}]} or a list
  POST /v1/systemone    -> Jev-style {"state", "questions": {...}} with noul / choice / score questions

Single-threaded on purpose: MLX/Metal work stays on the main thread. Bound to 127.0.0.1 only;
there is no authentication, so don't expose the port.
"""
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import __version__
from .config import HOST, MODEL, PID_FILE, PORT, REVISION, model_bits
from .rows import jev_answer, jev_to_rows

MAX_BODY = 4 * 1024 * 1024
MAX_TOKENS = int(os.environ.get("DECIDE_MAX_TOKENS", "4096"))


def make_handler(run, precision):
    class Handler(BaseHTTPRequestHandler):
        # One request at a time, so a client that connects and never sends a request must not
        # hold the server: drop it after 5 seconds.
        timeout = 5

        def log_message(self, *args):
            pass  # never log request contents

        def send(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"status": "ready", "model": MODEL, "revision": REVISION, "backend": "mlx",
                                       "precision": precision, "version": __version__, "pid": os.getpid()})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/score", "/v1/systemone"):
                return self.send(404, {"error": "not found"})
            origin = self.headers.get("Origin")
            if origin:  # block browsers on other sites from using the local model
                return self.send(403, {"error": "cross-origin requests are disabled"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.send(413, {"error": "request body size is outside server limits"})
                request = json.loads(self.rfile.read(length))
                if self.path == "/score":
                    rows = request if isinstance(request, list) else [request]
                    results = [run(r) for r in rows]
                    return self.send(200, results if isinstance(request, list) else results[0])
                t0 = time.perf_counter()
                answers = {qid: jev_answer(kind, run(row)) for qid, kind, row in jev_to_rows(request)}
                return self.send(200, {"answers": answers, "model": MODEL,
                                       "metadata": {"backend": "mlx", "seconds": time.perf_counter() - t0}})
            except (ValueError, KeyError, TypeError) as e:
                return self.send(422, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                return self.send(500, {"error": f"scoring failed: {type(e).__name__}"})

    return Handler


def main():
    from .scorer import Scorer

    # json prompt, one option order: the configuration that scored best on JevBench (see bench/RESULTS.md)
    bits = model_bits()
    precision = f"{bits}-bit" if bits else "bf16"
    print(f"loading {MODEL} ({precision})", flush=True)
    scorer = Scorer.load(MODEL, REVISION, prompt="json", orders=int(os.environ.get("DECIDE_ORDERS", "1")),
                         max_tokens=MAX_TOKENS, bits=bits)

    def run(row):
        for key in ("state", "question", "options"):
            if key not in row:
                raise ValueError(f"row is missing {key!r}")
        return scorer.score(row)

    # compile the Metal kernels now so the first real request is fast
    run({"id": "warmup", "state": "warm up", "question": "Ready?",
         "options": [{"id": "yes", "description": "yes"}, {"id": "no", "description": "no"}]})

    server = HTTPServer((HOST, PORT), make_handler(run, precision))
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(json.dumps({"url": f"http://{HOST}:{PORT}", "model": MODEL, "backend": "mlx", "precision": precision}), flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()


if __name__ == "__main__":
    main()
