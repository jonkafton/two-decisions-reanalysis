#!/usr/bin/env python3
"""Prediction 4: draw the disputed verdicts that people will label.

The population is every (item, system) verdict on the 368 matched items on
which the published judge and the second judge (pass 1) disagree: 213 of the
1,104. Sixty are drawn with a fixed seed before any labelling and written to
preregistration/p4_sample.json. Only identifiers are written; the text the
annotators read is assembled later from the fetched source files
(p4_annotation.py packets), because the benchmark's dialogues cannot be
redistributed here.

Run from anywhere:  python3 preregistration/p4_draw.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from common import SYSTEMS, matched_items, published_verdicts, second_verdicts  # noqa: E402

SEED = 20260925
N = 60
OUT = os.path.join(HERE, "p4_sample.json")


def disputed_verdicts():
    """Sorted (sample_id, system) pairs where the two judges disagree."""
    j1, j2 = published_verdicts(), second_verdicts()
    pairs = [(sid, s) for sid in matched_items() for s in SYSTEMS]
    assert len(pairs) == 1104, len(pairs)
    missing = [p for p in pairs if p not in j1 or p not in j2]
    assert not missing, f"verdicts missing for {missing[:5]}"
    disputed = sorted(p for p in pairs if j1[p] != j2[p])
    first_only = sum(1 for p in disputed if j1[p] == 1)
    assert (len(disputed), first_only) == (213, 210), (len(disputed), first_only)
    return disputed


def main():
    disputed = disputed_verdicts()
    drawn = random.Random(SEED).sample(disputed, N)
    out = {
        "prediction": 4,
        "procedure": ("The 213 (sample_id, system) pairs on the 368 matched items where the "
                      "published judge and the second judge (pass 1) disagree, sorted by "
                      "(sample_id, system), then random.Random(20260925).sample(pairs, 60) "
                      "in Python 3. Listed in draw order."),
        "seed": SEED,
        "population": len(disputed),
        "drawn": N,
        "items": [{"n": i + 1, "sample_id": sid, "system": s} for i, (sid, s) in enumerate(drawn)],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    j1 = published_verdicts()
    passed_by_first = sum(1 for p in drawn if j1[p] == 1)
    print(f"population {len(disputed)} disputed verdicts; drew {N} with seed {SEED}")
    print(f"of the drawn: {passed_by_first} passed only by the published judge, "
          f"{N - passed_by_first} passed only by the second judge")
    print(f"wrote {os.path.relpath(OUT, os.getcwd())}")


if __name__ == "__main__":
    main()
