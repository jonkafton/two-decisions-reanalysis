#!/usr/bin/env python3
"""The rest of the paper's §3 figures under the published judge.

  - the time-gap buckets: what the source's bucketing function does to each
    stated gap, and how turn distance tracks the stated gap (Spearman)
  - accuracy by measured cue-trigger distance, and exact McNemar tests within
    and beyond the nearest band
  - the one-judge, no-complementarity model (Appendix C): its exact fit and the
    judge accuracy it implies
  - the self-managing design's prompt size at answer time, from the usage
    record in each released response, against the size of the dialogue
  - responses carrying tool calls and returns, and how the judge scored them

Run ./fetch_data.sh first. Needs numpy and scipy (pip install -r requirements.txt).
"""
import collections
import importlib.util
import re
import statistics

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, spearmanr

from common import (DISTANCE_BANDS, SYSTEMS, load_json, matched_items, mcnemar_exact, pool,
                    published_verdicts, source)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, source(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = pool()
J1 = published_verdicts()
M = matched_items()
metrics = load_module("metrics")
prepare = load_module("prepare_data")

# ── time-gap buckets ─────────────────────────────────────────────────────────
print("== time-gap buckets (all 401 pool items) ==")
buckets = collections.Counter()
misfiled = collections.Counter()
for p in P.values():
    gap = p["time_gap"]
    b = metrics._bucket_time_gap(gap)
    buckets[b] += 1
    days = prepare._parse_time_gap(gap)
    text = gap.lower()
    if b == "short_1_to_2_weeks" and days > 14:
        misfiled["gap of three to eight weeks filed as short"] += 1
    if b == "long_3_plus_months" and "month" in text and 30 <= days < 90:
        misfiled["one- or two-month gap filed as long"] += 1
    if b == "long_3_plus_months" and "month" in text and days == 0:
        misfiled["months-worded but no number (e.g. 'several months'), filed as long"] += 1
    if b not in ("short_1_to_2_weeks", "medium_1_to_3_months", "long_3_plus_months"):
        misfiled[f"filed as {b!r} ({'year' if 'year' in text else 'other'}-worded)"] += 1
print("bucket counts:", dict(buckets))
print("misfiled:", dict(misfiled))
examples = sorted({p['time_gap'] for p in P.values() if metrics._bucket_time_gap(p['time_gap']) == 'short_1_to_2_weeks' and prepare._parse_time_gap(p['time_gap']) > 14})[:4]
print("e.g. filed as short:", examples)

dist = {i: P[i]["trigger_position"] - P[i]["cue_position"] for i in M}
days = {i: prepare._parse_time_gap(P[i]["time_gap"]) for i in M}
rho = spearmanr([days[i] for i in M], [dist[i] for i in M]).statistic
print(f"turn distance, matched items: min {min(dist.values())}, median {statistics.median(dist.values())}, max {max(dist.values())}")
print(f"Spearman rho, stated gap in days vs turn distance (matched): {rho:.2f}")

# ── distance bands under the published judge ─────────────────────────────────
print("\n== accuracy by cue-trigger distance, published judge (extraction / self-managing / graph) ==")
for lo, hi in DISTANCE_BANDS:
    ids = [i for i in M if lo <= dist[i] <= hi]
    accs = " / ".join(f"{100 * sum(J1[(i, s)] for i in ids) / len(ids):.1f}" for s in SYSTEMS)
    print(f"  {lo}-{hi} turns (n = {len(ids)}): {accs}")
for label, ids in (("within 96 turns", [i for i in M if dist[i] <= 96]), ("beyond 96 turns", [i for i in M if dist[i] > 96])):
    tests = []
    for a, b in (("mem0", "letta"), ("mem0", "zep"), ("letta", "zep")):
        _, _, p = mcnemar_exact([J1[(i, a)] for i in ids], [J1[(i, b)] for i in ids])
        tests.append(f"{a}-{b} p = {p:.4f}")
    print(f"  {label} (n = {len(ids)}): exact McNemar " + ", ".join(tests))

# ── the one-judge, no-complementarity model ──────────────────────────────────
print("\n== one-judge no-complementarity model (paper Appendix C) ==")
per_item = collections.Counter(sum(J1[(i, s)] for s in SYSTEMS) for i in M)
O = np.array([per_item[k] for k in range(4)], dtype=float)
N = O.sum()
k = np.arange(4)


def expected(params):
    q, g, f = params
    return N * (q * binom.pmf(k, 3, 1 - g) + (1 - q) * binom.pmf(k, 3, f))


def chi2(params):
    e = expected(params)
    return float(((O - e) ** 2 / e).sum())


best = None
for q0 in np.linspace(0.05, 0.6, 12):
    for g0 in np.linspace(0.05, 0.6, 12):
        for f0 in np.linspace(0.01, 0.3, 8):
            r = minimize(chi2, [q0, g0, f0], bounds=[(1e-3, 1 - 1e-3)] * 3, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
q, g, f = best.x
print("observed items with 0/1/2/3 systems correct:", O.astype(int).tolist())
print(f"fit (3 parameters, 3 degrees of freedom, so exact): q = {q:.3f}, g = {g:.3f}, f = {f:.3f}, chi-square = {best.fun:.4f}")
print(f"implied judge accuracy per verdict: {q * (1 - g) + (1 - q) * (1 - f):.3f}")

# ── the self-managing design's prompt at answer time ─────────────────────────
print("\n== self-managing design: prompt size at answer time vs dialogue size (matched items) ==")
responses = {(r["sample_id"], r["system"]): r for r in load_json(source("responses.json"))}
prompt, dialogue = [], []
for i in M:
    m = re.search(r"prompt_tokens=(\d+)", responses[(i, "letta")]["response"])
    if not m:
        continue
    prompt.append(int(m.group(1)))
    dialogue.append(sum(len(t["text"]) + len(t["speaker"]) + 2 for t in P[i]["full_dialogue_history"]) / 4)
ratio = [a / b for a, b in zip(prompt, dialogue)]
print(f"responses with a usage record: {len(prompt)} of {len(M)}")
print(f"prompt tokens: median {statistics.median(prompt):.0f} (min {min(prompt)}, max {max(prompt)})")
print(f"dialogue size, characters / 4: median {statistics.median(dialogue):.0f}")
print(f"ratio: median {statistics.median(ratio):.2f} (min {min(ratio):.2f}, max {max(ratio):.2f});  "
      f"correlation with dialogue size {np.corrcoef(prompt, dialogue)[0, 1]:.2f}")
tool_rows = [i for i in M if "ToolReturnMessage" in responses[(i, "letta")]["response"]]
print(f"responses carrying tool calls and returns: {len(tool_rows)}; judged correct by the published judge: "
      f"{sum(J1[(i, 'letta')] for i in tool_rows)}")
