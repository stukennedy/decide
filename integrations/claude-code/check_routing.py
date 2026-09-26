"""Run the router over labelled tasks: python check_routing.py [tasks.jsonl ...]"""
import importlib.util
import json
import sys
from pathlib import Path

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("router", here / "route_subagent.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)

if not router.ensure_server():
    sys.exit("decide server isn't running and couldn't be started")
files = sys.argv[1:] or [here / "sanity_tasks.jsonl"]
rows = [json.loads(line) for f in files for line in open(f) if line.strip()]
hits = 0
for t in rows:
    model, reason, details = router.route(f"{t['description']}\n\n{t['prompt']}")
    got = model or "(unchanged)"
    hits += got == t["expect"]
    print(f"{'✓' if got == t['expect'] else '✗'} {t['description']:<28} expect {t['expect']:<7} "
          f"got {got:<11} {reason:<14} {json.dumps(details)}")
print(f"\n{hits}/{len(rows)} as expected")
