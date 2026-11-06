#!/usr/bin/env python3
"""Prediction 2 (H2): draw the LoCoMo factual questions, stratified by category.

The population is every question in LoCoMo's locomo10.json (snap-research/locomo at
the commit fetch_data.sh pins; 1,986 questions over the ten conversations the pool's
dialogues are stitched from) whose evidence resolves to at least one turn of its
conversation ("D<session>:<turn>", present in the conversation). Nine questions have
evidence in another form and are excluded from the population before the draw, so
every drawn question has a defined palace scope (the base through its last evidence
session).

Stratified by LoCoMo's five categories (1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop, 5 adversarial): the questions of each category are sorted by
identifier and `random.Random(SEED).sample(ids, K)` is drawn in category order
1..5 from one generator. K defaults to 25, so 125 questions: the registered minimum
is 100 and exclusions for failed lookups must not take the scored count under it.

Only identifiers are written (the conversation's sample_id and the question's index
in its qa list) with a sha256 of the question text, so the driver can check it reads
the same question; the text itself is CC BY-NC 4.0 and not redistributed here.

  python3 preregistration/h2/p2_draw.py [--per-category 25] [--seed 20261104]
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from common import load_json, source  # noqa: E402

SEED = 20261104
OUT = os.path.join(HERE, "p2_factual_sample.json")
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DIA = re.compile(r"^D(\d+):(\d+)$")


def population(locomo):
    """[(id, category)] for questions with at least one resolvable evidence turn."""
    out, dropped = [], []
    for conv in locomo:
        c = conv["conversation"]
        dia_ids = set()
        i = 1
        while f"session_{i}" in c:
            for t in c[f"session_{i}"]:
                if t.get("dia_id"):
                    dia_ids.add(t["dia_id"])
            i += 1
        for idx, q in enumerate(conv["qa"]):
            qid = f"{conv['sample_id']}#{idx}"
            ok = [e for e in q.get("evidence") or [] if DIA.match(str(e)) and str(e) in dia_ids]
            if ok:
                out.append((qid, int(q["category"]), q["question"]))
            else:
                dropped.append(qid)
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=25)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    import random
    locomo = load_json(source("locomo10.json"))
    pop, dropped = population(locomo)
    rng = random.Random(args.seed)
    drawn = []
    by_cat = {}
    for cat in sorted(CATEGORY):
        ids = sorted(q for q in pop if q[1] == cat)
        by_cat[cat] = len(ids)
        if len(ids) < args.per_category:
            sys.exit(f"category {cat} has {len(ids)} questions, fewer than {args.per_category}")
        for qid, c, text in rng.sample(ids, args.per_category):
            drawn.append({"n": len(drawn) + 1, "id": qid, "conversation": qid.split("#")[0], "qa_index": int(qid.split("#")[1]),
                          "category": c, "category_name": CATEGORY[c],
                          "question_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    out = {
        "prediction": 2,
        "procedure": ("Every question in locomo10.json whose evidence resolves to a turn of its conversation "
                      f"({len(pop)} of {len(pop) + len(dropped)}; {len(dropped)} dropped for evidence in another form), "
                      "grouped by LoCoMo category, each group sorted by identifier (conversation sample_id '#' index in qa), "
                      f"then random.Random({args.seed}).sample(group, {args.per_category}) in category order 1..5 from one generator. "
                      "Identifiers and a sha256 of each question's text only; the text is not redistributed."),
        "seed": args.seed,
        "population_by_category": {CATEGORY[c]: n for c, n in by_cat.items()},
        "dropped_no_resolvable_evidence": dropped,
        "per_category": args.per_category,
        "drawn": len(drawn),
        "items": drawn,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"population {len(pop)} questions ({len(dropped)} dropped); drew {len(drawn)} ({args.per_category} per category) with seed {args.seed}")
    print(f"wrote {os.path.relpath(OUT, os.getcwd())}")


if __name__ == "__main__":
    main()
