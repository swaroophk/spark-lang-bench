#!/usr/bin/env python3
"""
Parse Spark benchmark logs and produce a CSV + plots.

USAGE
-----
python3 scripts/parse_and_plot.py \
  --logs results/*.log \
  --out-csv results/bench.csv \
  --out-png results/bench.png

WHAT IT DOES
------------
- Reads any lines that start with 'RESULT ' from the given logs.
- Parses common fields:
    lang, rows, users, shuffle, adaptive,
    WorkloadA_ms,
    WorkloadB_ms or WorkloadB_py_ms,
    WorkloadB_pandas_ms (optional, Python only)
- Writes a tidy CSV (one row per RESULT line).
- Aggregates repeated runs with the same (lang, rows, users, shuffle, adaptive) using the median.
- Produces a grouped bar chart comparing per-language medians for:
    - WorkloadA (SQL-native)
    - WorkloadB (UDF-heavy; Python column prefers pandas UDF if present)
"""
import argparse
import csv
import glob
import math
import os
import re
from statistics import median
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

RESULT_PREFIX = "RESULT "

# Example lines:
# RESULT scala rows=20000000 users=200000 shuffle=600 adaptive=true WorkloadA_ms=12345 WorkloadB_ms=67890
# RESULT java rows=20000000 users=200000 shuffle=600 adaptive=true WorkloadA_ms=12001 WorkloadB_ms=70000
# RESULT python rows=20000000 users=200000 shuffle=600 adaptive=true WorkloadA_ms=14002 WorkloadB_py_ms=99000 WorkloadB_pandas_ms=31000

KV_RE = re.compile(r"(\w+)=([^\s]+)")

def parse_result_line(line: str) -> Dict[str, str]:
    # Expect the line to start with RESULT <lang> ...
    if not line.startswith(RESULT_PREFIX):
        return {}
    parts = line.strip().split()
    if len(parts) < 3:
        return {}
    lang = parts[1]
    kvs = dict(KV_RE.findall(line))
    # Normalize keys:
    out = {
        "lang": lang.lower(),
        "rows": kvs.get("rows"),
        "users": kvs.get("users"),
        "shuffle": kvs.get("shuffle"),
        "adaptive": kvs.get("adaptive"),
        "WorkloadA_ms": kvs.get("WorkloadA_ms"),
        # Prefer generic 'WorkloadB_ms' if present (scala/java),
        # else fall back to Python-specific keys.
        "WorkloadB_ms": kvs.get("WorkloadB_ms"),
        "WorkloadB_py_ms": kvs.get("WorkloadB_py_ms"),
        "WorkloadB_pandas_ms": kvs.get("WorkloadB_pandas_ms"),
    }
    return out

def to_int_or_none(x: str):
    try:
        return int(x)
    except Exception:
        return None

def write_csv(rows: List[Dict[str, str]], out_csv: str):
    fieldnames = [
        "lang", "rows", "users", "shuffle", "adaptive",
        "WorkloadA_ms", "WorkloadB_ms", "WorkloadB_py_ms", "WorkloadB_pandas_ms",
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def aggregate_median(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Collapse repeated runs (same lang, rows, users, shuffle, adaptive)
    by taking median of numeric metrics.
    """
    key_fields = ("lang", "rows", "users", "shuffle", "adaptive")
    buckets: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = {}
    for r in rows:
        k = tuple(r.get(kf, "") for kf in key_fields)
        buckets.setdefault(k, []).append(r)

    out: List[Dict[str, str]] = []
    for k, group in buckets.items():
        agg = {kf: kv for kf, kv in zip(key_fields, k)}
        for metric in ("WorkloadA_ms", "WorkloadB_ms", "WorkloadB_py_ms", "WorkloadB_pandas_ms"):
            vals = [to_int_or_none(g.get(metric) or "") for g in group]
            vals = [v for v in vals if v is not None]
            agg[metric] = str(median(vals)) if vals else ""
        out.append(agg)
    return out

def select_udf_column(row: Dict[str, str]) -> Tuple[str, int]:
    """
    Decide which UDF number to plot:
    - Scala/Java: WorkloadB_ms
    - Python: prefer WorkloadB_pandas_ms if present, else WorkloadB_py_ms
    Returns (metric_name, value_int or math.nan)
    """
    lang = row.get("lang", "").lower()
    if lang in ("scala", "java"):
        m = row.get("WorkloadB_ms")
        return ("WorkloadB_ms", int(m)) if m else ("WorkloadB_ms", math.nan)
    # Python:
    m = row.get("WorkloadB_pandas_ms") or row.get("WorkloadB_py_ms")
    name = "WorkloadB_pandas_ms" if row.get("WorkloadB_pandas_ms") else "WorkloadB_py_ms"
    return (name, int(m)) if m else (name, math.nan)

def plot_grouped(agg_rows: List[Dict[str, str]], out_png: str):
    """
    For EACH unique (rows, users, shuffle, adaptive) config,
    make a grouped bar of languages for WorkloadA and WorkloadB.
    If multiple configs exist, we plot only the most common one
    and print a hint.
    """
    if not agg_rows:
        print("No rows to plot.")
        return

    # Group by config, choose the largest group
    cfg_key = lambda r: (r["rows"], r["users"], r["shuffle"], r["adaptive"])
    from collections import defaultdict
    groups = defaultdict(list)
    for r in agg_rows:
        groups[cfg_key(r)].append(r)
    # Choose the config with most languages present.
    best_cfg, best_group = max(groups.items(), key=lambda kv: len(kv[1]))

    langs = ["scala", "java", "python"]
    # Build arrays:
    wlA = []
    wlB = []
    labels = []
    rows, users, shuffle, adaptive = best_cfg
    # Index rows by lang for fast lookup
    by_lang = {r["lang"]: r for r in best_group}
    for lg in langs:
        r = by_lang.get(lg)
        if r:
            a = r.get("WorkloadA_ms")
            a_val = int(a) if a else math.nan
            wlA.append(a_val)
            _, b_val = select_udf_column(r)
            wlB.append(b_val)
        else:
            wlA.append(math.nan)
            wlB.append(math.nan)
        labels.append(lg.capitalize())

    # Plot
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    x = range(len(langs))
    width = 0.35

    # Workload A
    plt.figure(figsize=(8, 5))
    plt.bar(x, wlA, width)
    plt.xticks(list(x), labels)
    plt.ylabel("Milliseconds (lower is better)")
    plt.title(f"Workload A (SQL) — rows={rows}, users={users}, shuffle={shuffle}, adaptive={adaptive}")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_A.png"))
    plt.close()

    # Workload B
    plt.figure(figsize=(8, 5))
    plt.bar(x, wlB, width)
    plt.xticks(list(x), labels)
    plt.ylabel("Milliseconds (lower is better)")
    plt.title(f"Workload B (UDF) — rows={rows}, users={users}, shuffle={shuffle}, adaptive={adaptive}")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_B.png"))
    plt.close()

    print(f"Saved plots to: {out_png.replace('.png', '_A.png')} and {out_png.replace('.png', '_B.png')}")
    if len(groups) > 1:
        print("Note: multiple benchmark configs found; plotted the one with the most entries:")
        print(f"  rows={rows} users={users} shuffle={shuffle} adaptive={adaptive}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=False, help="Glob(s) for log files, e.g. results/*.log")
    ap.add_argument("--out-csv", default="results/bench.csv")
    ap.add_argument("--out-png", default="results/bench.png")
    args = ap.parse_args()

    log_paths = []
    if args.logs:
        for g in args.logs:
            log_paths.extend(glob.glob(g))
    else:
        log_paths = glob.glob("results/*.log")

    if not log_paths:
        print("No log files found. Provide --logs or put logs under results/*.log")
        return

    rows: List[Dict[str, str]] = []
    for p in sorted(log_paths):
        with open(p, "r", errors="ignore") as f:
            for line in f:
                if line.startswith(RESULT_PREFIX):
                    rec = parse_result_line(line)
                    if rec:
                        rows.append(rec)

    if not rows:
        print("No RESULT lines found in provided logs.")
        return

    # Write raw CSV of all observed results
    write_csv(rows, args.out_csv)

    # Aggregate to medians per (lang, rows, users, shuffle, adaptive)
    agg_rows = aggregate_median(rows)
    write_csv(agg_rows, args.out_csv.replace(".csv", "_median.csv"))

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote median CSV: {args.out_csv.replace('.csv','_median.csv')}")

    # Plot from the median-aggregated rows
    plot_grouped(agg_rows, args.out_png)

if __name__ == "__main__":
    main()
