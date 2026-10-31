"""Shared paths, loaders and small statistics for the reanalysis scripts."""
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.environ.get("SOURCE_DIR", os.path.join(ROOT, "data", "source"))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SYSTEMS = ("mem0", "letta", "zep")
NAME = {"mem0": "extraction (Mem0)", "letta": "self-managing (Letta)", "zep": "temporal graph (Zep)"}
# Cue-trigger distance bands, in turns, used throughout the paper.
DISTANCE_BANDS = ((1, 96), (97, 184), (185, 273), (274, 393), (394, 690))


def source(name):
    path = os.path.join(SOURCE, name)
    if not os.path.exists(path):
        sys.exit(f"missing {path}\nRun ./fetch_data.sh first (it downloads the companion repository's files).")
    return path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def published_verdicts():
    """{(sample_id, system): 1 if correct else 0} for the released verdicts."""
    return {(r["sample_id"], r["system"]): int(r["label"] == "correct")
            for r in load_json(source("judge_results.json"))}


def second_verdicts(path=None, which_pass=1):
    """{(sample_id, system): 1/0} for our second judge; pass 2 is the retest."""
    path = path or os.path.join(DATA, "second_judge_verdicts.jsonl")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("pass") == which_pass and r.get("label") in ("correct", "wrong"):
                out[(r["sample_id"], r["system"])] = int(r["label"] == "correct")
    return out


def matched_items():
    return load_json(os.path.join(DATA, "matched_items.json"))


def pool():
    return {p["sample_id"]: p for p in load_json(source("pool.json"))}


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1)


def binom_two_sided(k, n):
    """Exact two-sided binomial p-value at p = 0.5 (the exact McNemar test)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_exact(x, y):
    b = sum(1 for a, c in zip(x, y) if a == 1 and c == 0)
    c = sum(1 for a, d in zip(x, y) if a == 0 and d == 1)
    return b, c, binom_two_sided(b, b + c)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def cochran_q(rows):
    """rows: list of per-item tuples of 0/1 across k systems. Returns (Q, p)."""
    k = len(rows[0])
    col = [sum(r[j] for r in rows) for j in range(k)]
    row = [sum(r) for r in rows]
    total = sum(row)
    num = (k - 1) * (k * sum(c * c for c in col) - total * total)
    den = k * total - sum(t * t for t in row)
    q = num / den if den else 0.0
    # chi-square survival function with k - 1 = 2 degrees of freedom is exp(-q / 2)
    p = math.exp(-q / 2) if k == 3 else None
    return round(q, 2), p


def pct(x):
    return f"{100 * x:.1f}"


def by_item(verdicts, items):
    return {i: tuple(verdicts[(i, s)] for s in SYSTEMS) for i in items}


def ensure_results():
    os.makedirs(RESULTS, exist_ok=True)


def group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out
