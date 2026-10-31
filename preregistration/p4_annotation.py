#!/usr/bin/env python3
"""Prediction 4: build the annotators' packets, then score their labels.

  packets  Writes one packet per annotator (A and B) to preregistration/packets/
           (git-ignored, because it holds the benchmark's text): the 60 drawn
           items in an order shuffled separately for each annotator, each shown
           as the judges saw it (Evidence (Cue), Question (Trigger), Model
           Prediction), with no judge verdict and no system name. Also writes a
           blank labels CSV per annotator and the key that maps item numbers
           back to (sample_id, system); the key stays with the author.
  score    Reads the two labels CSVs and applies the registered decision rule.

Registered rule. On the items both annotators label identically (k of them),
let a be the number whose shared label matches the second judge's verdict.
  supported     3a >= 2k (they side with the second judge on at least two thirds)
  failed        2(k - a) >= k (they side with the published judge on at least half)
  inconclusive  otherwise, or if k < 20
Agreement between the annotators is reported as raw agreement and Cohen's kappa.

  python3 preregistration/p4_annotation.py packets
  python3 preregistration/p4_annotation.py score --a labels_A.csv --b labels_B.csv
"""
import argparse
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from common import RESULTS, pool, published_verdicts, second_verdicts  # noqa: E402

SAMPLE = os.path.join(HERE, "p4_sample.json")
INSTRUCTIONS = os.path.join(HERE, "p4_instructions.md")
SEED = 20260925
ORDER_SEED = {"A": SEED + 1, "B": SEED + 2}
MIN_AGREED = 20
LABELS = ("correct", "wrong")


def sample_pairs():
    d = json.load(open(SAMPLE, encoding="utf-8"))
    assert d["drawn"] == len(d["items"]) == 60
    return [(it["sample_id"], it["system"]) for it in d["items"]]


def responses():
    from common import load_json, source
    return {(r["sample_id"], r["system"]): r["response"] for r in load_json(source("responses.json"))}


def packets(out_dir):
    pairs, items, resp = sample_pairs(), pool(), responses()
    os.makedirs(out_dir, exist_ok=True)
    brief = open(INSTRUCTIONS, encoding="utf-8").read().strip()
    for who, seed in ORDER_SEED.items():
        order = list(range(len(pairs)))
        random.Random(seed).shuffle(order)
        key, parts = {}, [brief, ""]
        for n, idx in enumerate(order, start=1):
            sid, system = pairs[idx]
            key[n] = {"sample_id": sid, "system": system}
            p = items[sid]
            parts += [f"## Item {n}", "", "**Evidence (Cue):**", "", p["cue_dialogue"], "",
                      "**Question (Trigger):**", "", p["trigger_query"], "",
                      "**Model Prediction:**", "", resp[(sid, system)], "", "---", ""]
        with open(os.path.join(out_dir, f"packet_{who}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        with open(os.path.join(out_dir, f"labels_{who}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "label"])
            for n in range(1, len(pairs) + 1):
                w.writerow([n, ""])
        with open(os.path.join(out_dir, f"key_{who}.json"), "w", encoding="utf-8") as f:
            json.dump(key, f, indent=1)
    print(f"wrote packets, blank label files and keys for annotators A and B to {out_dir}")


def read_labels(path, key):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().lower()
            if label in LABELS:
                k = key[str(int(row["item"]))]
                out[(k["sample_id"], k["system"])] = label
    return out


def kappa(pairs, la, lb):
    both = [p for p in pairs if p in la and p in lb]
    n = len(both)
    if n == 0:
        return float("nan"), 0.0
    agree = sum(la[p] == lb[p] for p in both) / n
    pa = sum(la[p] == "correct" for p in both) / n
    pb = sum(lb[p] == "correct" for p in both) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (agree - pe) / (1 - pe) if pe < 1 else float("nan"), agree


def decide(k, a):
    if k < MIN_AGREED:
        return "inconclusive"
    if 3 * a >= 2 * k:
        return "supported"
    if 2 * (k - a) >= k:
        return "failed"
    return "inconclusive"


def score(path_a, path_b, key_dir, out_path):
    pairs = sample_pairs()
    key_a = json.load(open(os.path.join(key_dir, "key_A.json"), encoding="utf-8"))
    key_b = json.load(open(os.path.join(key_dir, "key_B.json"), encoding="utf-8"))
    la, lb = read_labels(path_a, key_a), read_labels(path_b, key_b)
    j2 = second_verdicts()
    j1 = published_verdicts()
    agreed = [p for p in pairs if p in la and p in lb and la[p] == lb[p]]
    k = len(agreed)
    a = sum(1 for p in agreed if (la[p] == "correct") == bool(j2[p]))
    assert all(j1[p] != j2[p] for p in pairs), "a drawn verdict is not disputed"
    kap, raw = kappa(pairs, la, lb)
    result = {
        "prediction": 4,
        "labelled_by_A": len(la), "labelled_by_B": len(lb),
        "raw_agreement": round(raw, 4), "cohen_kappa": round(kap, 4),
        "agreed_items_k": k,
        "agreed_with_second_judge_a": a,
        "agreed_with_published_judge": k - a,
        "decision": decide(k, a),
        "rule": "supported if 3a >= 2k; failed if 2(k-a) >= k; otherwise, or if k < 20, inconclusive",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packets")
    p.add_argument("--out", default=os.path.join(HERE, "packets"))
    s = sub.add_parser("score")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--keys", default=os.path.join(HERE, "packets"))
    s.add_argument("--out", default=os.path.join(RESULTS, "p4_result.json"))
    args = ap.parse_args()
    if args.cmd == "packets":
        packets(args.out)
    else:
        score(args.a, args.b, args.keys, args.out)


if __name__ == "__main__":
    main()
