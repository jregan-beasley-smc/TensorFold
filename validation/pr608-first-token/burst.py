"""Retain raw request evidence from TensorFold's existing concurrent client."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


def main() -> None:
    """Run one bounded prompt/concurrency cell, retaining every request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", choices=("code", "chat"), default="code")
    parser.add_argument("--streams", type=int, choices=(1, 4, 16), default=16)
    parser.add_argument("--temperature", type=float, choices=(0.0, 1.0), default=0.0)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    if args.tokens < 4 or args.reps < 1:
        parser.error("tokens must be at least 4 and reps at least 1")
    client_path = args.tools.resolve() / "bench_concurrent.py"
    sys.path.insert(0, str(client_path.parent))
    bench = importlib.import_module("bench_concurrent")
    if Path(bench.__file__).resolve() != client_path:
        raise RuntimeError("unexpected benchmark module")
    prompt = next(item for item in bench.PROMPTS if item["name"] == args.prompt)
    specs = [(prompt, 1234 + i if args.temperature else 1234)
             for i in range(args.streams)]
    with args.output.open("x", encoding="utf-8") as output:
        def save(record: dict[str, Any]) -> None:
            """Flush each complete phase so an interrupted run retains evidence."""
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()

        save({"kind": "identity", "label": args.label,
              "client_sha256": hashlib.sha256(client_path.read_bytes()).hexdigest(),
              "evidence_sha256": hashlib.sha256(
                  client_path.with_name("bench_evidence.py").read_bytes()).hexdigest(),
              "streams": args.streams, "temperature": args.temperature,
              "tokens": args.tokens, "reps": args.reps,
              "prompt": prompt, "seeds": [seed for _, seed in specs],
              "first_use_scope": "first burst in this invocation; server freshness recorded separately",
              "started_unix": time.time()})

        def valid_run(run: dict[str, Any]) -> bool:
            """Require complete equal work and verified cold-cache evidence."""
            return (not run.get("error") and run.get("complete") is True
                    and run.get("tokens") == args.tokens
                    and run.get("cache_state") == "cold"
                    and run.get("cached_tokens") == 0
                    and run.get("ttft_s") is not None
                    and run.get("decode_tps") is not None
                    and not run.get("unmeasured"))
        all_runs: list[tuple[int, list[dict[str, Any]]]] = []
        for index in range(args.reps + 1):
            phase = "first_use" if index == 0 else "warm"
            runs = bench.together(args.base, args.model, specs, args.tokens,
                                  args.temperature)
            save({"kind": "burst", "phase": phase, "index": index, "runs": runs})
            valid = all(valid_run(run) for run in runs)
            if len(runs) != args.streams or not valid:
                raise RuntimeError("burst failed request, work, or cold-cache checks")
            times = [run["ttft_s"] for run in runs]
            print(json.dumps({"phase": phase, "index": index,
                              "median_ttft_s": statistics.median(times),
                              "max_ttft_s": max(times),
                              "client_send_spread_s": max(r["sent"] for r in runs)
                              - min(r["sent"] for r in runs)}), flush=True)
            all_runs.append((index, runs))
        references: dict[int, dict[str, Any]] = {}
        for index, (item, seed) in enumerate(specs):
            if seed not in references:
                drafted = bench.stream(args.base, args.model, item, args.tokens,
                                       args.temperature, seed)
                plain = bench.stream(args.base, args.model, item, args.tokens,
                                     args.temperature, seed, draft=False)
                references[seed] = drafted
                save({"kind": "solo", "seed": seed, "drafted": drafted,
                      "plain": plain, "equal": bench.token_equal(drafted, plain)})
                if not (valid_run(drafted) and valid_run(plain)):
                    raise RuntimeError("solo work or cold-cache evidence failed")
                if bench.token_equal(drafted, plain) is not True:
                    raise RuntimeError("drafted/plain parity failed or unavailable")
            for burst_index, runs in all_runs:
                equal = bench.token_equal(runs[index], references[seed])
                save({"kind": "concurrent_vs_solo", "burst": burst_index,
                      "stream": index, "equal": equal})
                if equal is not True:
                    raise RuntimeError("concurrent/solo parity failed or unavailable")
        save({"kind": "complete"})


if __name__ == "__main__":
    main()
