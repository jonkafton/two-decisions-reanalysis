# Registered predictions

The paper's four predictions were registered on OSF before any of them was tested. The registration's address is added here once it is public.

This folder holds everything the registration fixes for predictions 3 and 4. Both run on the 368 matched items and need no new system. Predictions 1 and 2 need studies that do not exist yet; each will get its own registration before its data are collected.

| File | What it is |
|---|---|
| `p3_trigger_only.py` | Prediction 3's run: a model given only the trigger, scored by the published judge and by the second judge. `--stub` runs the whole pipeline with stand-in models and no network |
| `p3_score.py` | Prediction 3's decision rule, with its thresholds derived from the released verdicts |
| `requirements-p3.txt` | The two packages the real prediction 3 run needs; the published judge's own code imports them. Versions as pinned by the companion repository |
| `p4_draw.py` | Prediction 4's draw: 60 of the 213 disputed verdicts, seed 20260925. Run before registration; its output is `p4_sample.json` |
| `p4_sample.json` | The 60 drawn verdicts, by identifier only |
| `p4_instructions.md` | What the two annotators are told |
| `p4_annotation.py` | Builds each annotator's packet, in a separately shuffled order with no verdicts and no system names, and applies prediction 4's decision rule to their labels |

## Order of work

1. Nothing here runs against a real model, or reaches an annotator, before the registration is approved. A `--stub` run makes no calls and is the only exception.
2. `./fetch_data.sh`. It also fetches the published judge's `evaluation/judge.py` and `config.py`, hash-checked, so that prediction 3 is scored by that code and not by a copy of it.
3. Prediction 3: `pip install -r preregistration/requirements-p3.txt`, then `OPENAI_API_KEY=... GOOGLE_API_KEY=... python3 preregistration/p3_trigger_only.py`, then `python3 preregistration/p3_score.py`. The second judge runs through the Claude Code CLI, as in `scripts/second_judge_run.py`.
4. Prediction 4: `python3 preregistration/p4_annotation.py packets`. Give each annotator only their own packet and blank labels file; the keys stay with the author. Then `python3 preregistration/p4_annotation.py score --a labels_A.csv --b labels_B.csv`.

The packets hold the benchmark's text, so they are written to `preregistration/packets/`, which is git-ignored, and are never published here.
