#!/usr/bin/env python3
"""The published judge's verdicts, item by item (paper §3, Table 1).

Prints:
  - the verdict rows and identifiers
  - the 33 identifiers whose relation type or time gap differ between systems
    (the protocol deviation), and checks the matched set against
    data/matched_items.json
  - accuracy per system on all identifiers and on the 368 matched items
  - Cochran's Q and Holm-adjusted exact McNemar tests, on all identifiers
    (the whitepaper's figures) and on the matched items
  - how many matched items have 0 / 1 / 2 / 3 systems judged correct
  - the union, and the items solved by a non-leader but not the leader
  - Fleiss' kappa between the three systems' verdicts
  - the share of judge rationales cut off after the label

Run ./fetch_data.sh first.
"""
import collections

from common import (SYSTEMS, cochran_q, holm, load_json, matched_items, mcnemar_exact,
                    source, wilson)

rows = load_json(source("judge_results.json"))
print("verdict rows:", len(rows), dict(collections.Counter(r["system"] for r in rows)))

by_id = collections.defaultdict(dict)
for r in rows:
    by_id[r["sample_id"]][r["system"]] = r
ids = sorted(by_id)
print("identifiers:", len(ids))

mismatched, differs = [], collections.Counter()
for i in ids:
    labels = {s: (by_id[i][s]["relation_type"], by_id[i][s]["time_gap"]) for s in by_id[i]}
    if len(set(labels.values())) > 1:
        mismatched.append(i)
        for s in ("mem0", "letta"):
            if labels[s] != labels["zep"]:
                differs[s] += 1
print("identifiers with mismatched relation type / time gap:", len(mismatched), dict(differs))
matched = [i for i in ids if i not in set(mismatched)]
assert matched == sorted(matched_items()), "matched set differs from data/matched_items.json"
print("matched items:", len(matched), "(equal to data/matched_items.json)")


def correct(i, s):
    return int(by_id[i][s]["label"] == "correct")


for s in SYSTEMS:
    a = sum(correct(i, s) for i in ids)
    m = sum(correct(i, s) for i in matched)
    print(f"{s:6s} all: {a}/{len(ids)} = {wilson(a, len(ids))}   matched: {m}/{len(matched)} = {wilson(m, len(matched))}")

for label, items in (("all identifiers", ids), ("matched items", matched)):
    q, p = cochran_q([tuple(correct(i, s) for s in SYSTEMS) for i in items])
    pairs = (("mem0", "letta"), ("mem0", "zep"), ("letta", "zep"))
    raw = [mcnemar_exact([correct(i, a) for i in items], [correct(i, b) for i in items]) for a, b in pairs]
    adj = holm([r[2] for r in raw])
    print(f"{label}: Cochran's Q = {q} (p = {p:.3f});  Holm-adjusted exact McNemar: "
          + ", ".join(f"{a}-{b} {x:.3f}" for (a, b), x in zip(pairs, adj)))

dist = collections.Counter(sum(correct(i, s) for s in SYSTEMS) for i in matched)
print("matched items with 0/1/2/3 systems correct:", [dist[k] for k in range(4)])

union = sum(any(correct(i, s) for s in SYSTEMS) for i in matched)
leader = max(SYSTEMS, key=lambda s: sum(correct(i, s) for i in matched))
non_leader = sum(any(correct(i, s) for s in SYSTEMS) and not correct(i, leader) for i in matched)
print(f"union: {union}/{len(matched)} = {wilson(union, len(matched))};  leader: {leader};  "
      f"solved by another but not the leader: {non_leader} = {wilson(non_leader, len(matched))}")

n, N = 3, len(matched)
p_i, tally = [], [0, 0]
for i in matched:
    c1 = sum(correct(i, s) for s in SYSTEMS)
    c0 = n - c1
    p_i.append((c1 * c1 + c0 * c0 - n) / (n * (n - 1)))
    tally[0] += c0
    tally[1] += c1
p_bar = sum(p_i) / N
p_e = sum((t / (N * n)) ** 2 for t in tally)
print("Fleiss' kappa on matched items:", round((p_bar - p_e) / (1 - p_e), 3))

truncated = sum(r["reason"].startswith("Partial judge JSON") for r in rows)
print(f"judge rationales truncated after the label: {truncated}/{len(rows)} = {100 * truncated / len(rows):.0f}%")
