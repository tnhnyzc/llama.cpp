#!/usr/bin/env python3
"""Small Step MTP server benchmark helper.

The goal is not to replace a full benchmark harness. It just runs a fixed prompt
set against llama-server's /completion endpoint so p/q, temperature, and draft
depth comparisons stay repeatable across local experiments.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


PROMPTS = [
    "Write a concise technical paragraph explaining speculative decoding.",
    "Implement a Python function that returns the first ten Fibonacci numbers, then explain it briefly.",
    "Give a short, careful answer: why can MoE inference be slower than expected for small batch sizes?",
    "Continue this sentence in a natural assistant style: The most important thing to remember about benchmarking language models is",
]


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_path(obj: dict, *path: str):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_bool_list(value: str) -> list[bool]:
    out = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"1", "true", "on", "yes"}:
            out.append(True)
        elif item in {"0", "false", "off", "no"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid bool value: {item}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:10009")
    parser.add_argument("--draft", type=int, default=1)
    parser.add_argument("--temps", default="0.6,1.0")
    parser.add_argument("--pq", type=as_bool_list, default=as_bool_list("false,true"))
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache-prompt", action="store_true")
    args = parser.parse_args()

    temps = [float(item) for item in args.temps.split(",")]

    rows = []
    for temp in temps:
        for pq_accept in args.pq:
            for i, prompt in enumerate(PROMPTS):
                payload = {
                    "prompt": prompt,
                    "n_predict": args.n_predict,
                    "temperature": temp,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "cache_prompt": args.cache_prompt,
                    "speculative.n_max": args.draft,
                    "speculative.pq_accept": pq_accept,
                }
                t0 = time.monotonic()
                try:
                    resp = post_json(args.url, payload)
                except urllib.error.URLError as exc:
                    print(f"request failed for temp={temp} pq={pq_accept}: {exc}", file=sys.stderr)
                    return 2
                wall = time.monotonic() - t0

                timings = resp.get("timings", {})
                pred_n = timings.get("predicted_n") or resp.get("tokens_predicted") or 0
                tok_s = timings.get("predicted_per_second")
                if tok_s is None and pred_n:
                    tok_s = pred_n / wall

                spec = resp.get("speculative", {})
                drafted = (
                    spec.get("drafted")
                    or spec.get("n_drafted")
                    or get_path(resp, "speculative", "draft", "generated")
                )
                accepted = (
                    spec.get("accepted")
                    or spec.get("n_accepted")
                    or get_path(resp, "speculative", "draft", "accepted")
                )
                acc_rate = (accepted / drafted) if drafted else None

                rows.append({
                    "prompt": i,
                    "temp": temp,
                    "pq": pq_accept,
                    "tok_s": tok_s,
                    "drafted": drafted,
                    "accepted": accepted,
                    "acc_rate": acc_rate,
                    "pred_n": pred_n,
                    "wall_s": wall,
                })

                acc = "-" if acc_rate is None else f"{acc_rate:.4f}"
                print(
                    f"prompt={i} temp={temp:g} pq={str(pq_accept).lower()} "
                    f"tok/s={tok_s:.2f} acc={acc} pred_n={pred_n}"
                )

    print("\nsummary")
    for temp in temps:
        for pq_accept in args.pq:
            subset = [r for r in rows if r["temp"] == temp and r["pq"] == pq_accept]
            tok_vals = [r["tok_s"] for r in subset if r["tok_s"] is not None]
            acc_vals = [r["acc_rate"] for r in subset if r["acc_rate"] is not None]
            tok_avg = statistics.mean(tok_vals) if tok_vals else float("nan")
            if acc_vals:
                acc_avg = f"{statistics.mean(acc_vals):.4f}"
            else:
                acc_avg = "not reported by server"
            print(f"temp={temp:g} pq={str(pq_accept).lower()} avg_tok/s={tok_avg:.2f} avg_acc={acc_avg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
