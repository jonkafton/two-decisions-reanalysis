# two-decisions-reanalysis

Code and data for **Two Decisions, Not Three Architectures: A Reanalysis of Cognitive Memory in LLM Agents** (Jon Kafton, 2026).

- Reading version: https://1presence.com/blog/two-decisions-not-three-architectures
- PDF, the version of record: https://1presence.com/papers/two-decisions-not-three-architectures-v4.pdf

The paper reanalyses the item-level results that the appliedAI Initiative released with its 2026 whitepaper *Agentic Memory Realized*. The whitepaper is a matched comparison of three agent-memory systems (Mem0, Letta and Zep) on the cognitive track of the LoCoMo-Plus benchmark. The paper also re-scores every matched response with a second judge, from a model family different from both the published judge and the answering backbone.

This repository holds the second judge's verdicts, and every script needed to reproduce the figures in the paper's §3 and Appendices A–C from them and from the whitepaper's released files. It also holds the registered tests of the paper's four predictions, with their results.

## Findings reproduced here

- **A protocol deviation.** For 33 of the 401 item identifiers, systems were scored on different dialogues under the same identifier. 368 items remain matched.
- **The published figures, under the published judge.** On the matched items: 18.2%, 24.7% and 22.3% for Mem0, Letta and Zep; Cochran's Q = 7.11; a union of 40.8%.
- **The second judge.** It runs the same prompt on the same responses and agrees with the published judge only slightly (Cohen's κ = 0.18).
  - Accuracy is 4.3%, 2.4% and 2.2%, with no significant difference (Q = 3.68, p = 0.16).
  - The union is 8.4%.
  - The judges part on replies that are consistent with the cue but could have been written from the trigger alone.
- **The time-gap buckets are mislabelled,** and the distance curve depends on the judge.
- **The self-managing system answered with a prompt the size of the whole dialogue.** This comes from the usage records in its released responses: median about 21,100 tokens against about 21,200.

## Registered predictions

The paper ends on four predictions. They were registered on OSF on 3 November 2026, before any of them was tested (https://osf.io/xufd9). Predictions 3 and 4 run on the 368 matched items. Predictions 1 and 2 needed a new study, registered on 6 November 2026 before any item ran (https://osf.io/xf3gk). Each outcome below follows the decision rule fixed in its registration.

- **Prediction 1: inconclusive.** A bounded, curated standing state raises goal and state accuracy more than causal and value accuracy. On 401 items, none excluded, a verdict counting only when both judges pass:
  - the goal-and-state gain exceeded the causal-and-value gain by 18.6 points (bootstrap 95% interval 9.6 to 27.6), against a bar of 5;
  - false alarms on negative controls rose by 1.0 point (8 of 802 against 0 of 802), against a ceiling of 5;
  - the third condition was not met. The gain was to be larger on the nearest distance quintile than on the farthest, and it rose with distance instead: 16.0, 23.5, 28.4, 23.5 and 32.5 points across the five quintiles.

  `results/xf3gk/h1_result.json`.
- **Prediction 2: supported.** Improving query planning raises factual accuracy more than cognitive accuracy. With the planner on, factual accuracy rose by 9.6 points (46 of 125 against 34) and cognitive accuracy fell by 5.5 (117 of 401 against 139). The difference is 15.1 points (bootstrap 95% interval 7.1 to 23.4), against a bar of 5. `results/xf3gk/merged/p2_result.json`.
- **Prediction 3: supported.** A model given only the trigger, with no dialogue and no memory, scores at least half as well as the memory systems under the published judge. It scored 92 of 368 (25.0%) under the published judge, against a bar of 10.9%, and 5 of 368 (1.4%) under the second judge, under the 2% at which the prediction would have failed. `results/p3_result.json`.
- **Prediction 4: supported.** People side with the second judge on the replies the judges dispute. Two annotators, blind to both judges' verdicts and to the systems, labelled 60 of the 213 disputed verdicts. They gave the same label on 44 (raw agreement 0.73, Cohen's κ = 0.26), and on 39 of those 44 it was the second judge's, against a bar of two thirds. `results/p4_result.json`.

## Reproduce

```bash
./fetch_data.sh                          # the companion repository's files, pinned to a commit and checksummed
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd scripts
python3 verify_item_level.py             # published judge: the deviation, Table 1, the union, Fleiss' kappa
python3 published_extras.py              # buckets, distance bands, the Appendix C model, prompt sizes
python3 second_judge_analysis.py         # second judge: Table 1 rows, agreement, union, bands, latent-class fits
for r in 0.02 0.03 0.05; do RARE=$r python3 second_judge_lexical.py; done   # the judge-free check
```

`verify_item_level.py` needs only the Python standard library. The other scripts need numpy and scipy. `second_judge_analysis.py` takes a few minutes, because of its bootstrap draws (`BOOT=200`, `PBOOT=100`, as released). It rewrites `results/second_judge_analysis.json`, and the fits are seeded, so the file should come back unchanged.

**Expected output, in brief**
- **`verify_item_level.py`:**
  - 1,203 rows, 401 identifiers, 33 mismatched (Mem0 33, Letta 29), 368 matched;
  - 18.2 / 24.7 / 22.3;
  - Q = 7.14 on all identifiers, with Holm-adjusted exact McNemar 0.040, 0.171 and 0.466, which are the whitepaper's figures;
  - Q = 7.11 on the matched items;
  - 218 / 86 / 38 / 26 items with zero to three systems correct;
  - union 150 (40.8%), Fleiss' κ 0.34, 87% of judge rationales truncated.
- **`published_extras.py`:**
  - 58 gaps of three to eight weeks filed as "1–2 weeks", and 118 one- or two-month gaps filed as "3+ months";
  - Spearman ρ = 0.94;
  - distance bands 15.7 / 32.9 / 37.1 within 96 turns;
  - a model fit of q = 0.184, g = 0.276, f = 0.103, exact because the model is saturated;
  - an implied judge accuracy of 0.865.
- **`second_judge_analysis.py`:**
  - 4.3 / 2.4 / 2.2;
  - κ = 0.177, and 30 / 210 / 3 / 861 cells;
  - union 8.4%;
  - retest 150 of 150;
  - replication of the published judge's passes 19.8% (one system), 10.5% (two) and 6.4% (all three).
- **`second_judge_lexical.py`:** 76.7 / 33.3 / 14.2% of cells carry a rare cue word, and the mixture share is 24.2%, 30.7% and 29.3% at the three thresholds.

## How the second judge was run

- **Prompt.** The benchmark's own judge prompt, `COGNITIVE_AWARENESS_PROMPT`, as locked in the companion repository (sha256 `7174feb3…`, asserted before every call). It was filled with the pool item's cue and trigger and the system's released response, and nothing else was sent: no system prompt and no tools.
- **Model.** `claude-opus-5-5`, through the Claude Code CLI 2.1.280, on 25 September 2026, six calls in parallel:

  ```bash
  claude --print --output-format json --model opus --effort medium \
    --tools "" --strict-mcp-config --setting-sources project \
    --no-session-persistence --system-prompt ""   # run from an empty directory
  ```

  The published run used `gpt-4o-mini` as the answering backbone and `gemini-2.5-flash` as the judge (the companion repository's `run_meta.json`).
- **Temperature.** The CLI cannot set temperature, so a seeded random 150 of the verdicts were judged a second time. All 150 matched.
- **Parser.** The original judge's parser, without its fallback that turned an unparseable reply into "wrong". Every reply parsed.
- **Rows.** 1,104 verdicts on the 368 matched items, plus the 150 retests: 1,254 calls, with no failures. The 33 mismatched identifiers are excluded, because the published judge's inputs for them cannot be reconstructed from the release.

`scripts/second_judge_run.py` is the runner. It writes to `results/rerun_verdicts.jsonl` (or `$VERDICTS_OUT`), never over the released data, and it resumes where it stopped.

## Files

| Path | What it is |
|---|---|
| `data/second_judge_verdicts.jsonl` | The 1,254 verdicts. Fields: `sample_id`, `system`, `pass` (1, or 2 for the retest), `label`, `reason`, `model`, token counts (`in`, `cache_w`, `cache_r`, `out`), `cost_usd_list` (the CLI's list-price estimate), `api_ms`, `wall_s`, `ts` |
| `data/matched_items.json` | The 368 identifiers all three systems answered on the same dialogue. `verify_item_level.py` re-derives the list and checks it |
| `data/SOURCE_SHA256` | Checksums of the files `fetch_data.sh` downloads |
| `results/second_judge_analysis.json` | The output the paper's second-judge figures were read from |
| `results/p3_result.json` | Prediction 3's registered score: the trigger-only counts and Wilson intervals under both judges, the thresholds, the decision, and an exploratory split by relation type |
| `results/p3_replies.jsonl` | The 368 trigger-only replies. Fields: `sample_id`, `reply`, `model`, `system_fingerprint`, `finish_reason`, `tokens_in`, `tokens_out`, `ts` |
| `results/p3_verdicts.jsonl` | The 736 verdicts on those replies, one from each judge. Fields: `sample_id`, `judge` (`published` or `second`), `model`, `label`, `reason`, `raw`, `ts` |
| `results/p3_meta.jsonl` | Prediction 3's run record: the protocol lock, the backbone and its settings, the system prompt, the second judge's model and effort, and the CLI version |
| `results/p4_result.json` | Prediction 4's registered score: both annotators' agreement (raw and Cohen's kappa), the shared labels, how many sided with each judge, and the decision |
| `results/p4/` | Prediction 4's labels: each annotator's `labels_*.csv` by packet item number, the `key_*.json` mapping item numbers to (`sample_id`, `system`), and `notes_*.md`, the remarks each annotator offered with their labels, verbatim. `labels_compared.csv` joins the two by verdict with both judges' labels and whether the published judge's output was cut at its token cap. No benchmark text; the packets themselves are not released |
| `results/xf3gk/` | The per-item record of the registered run of predictions 1 and 2 (OSF registration xf3gk): the benchmarks' text replaced by SHA-256 and length, the pipeline's outputs in full, both scorers' outputs, `SHA256SUMS` and an OpenTimestamps proof; its `README.md` explains the fields. The same set is on OSF (osf.io/ymc78) and in the `run-xf3gk` release |
| `scripts/common.py` | Paths, loaders and small statistics shared by the scripts |
| `preregistration/` | The registered tests of predictions 3 and 4: runners, decision rules, the drawn sample and the annotators' instructions. See `preregistration/README.md` |
| `preregistration/h2/` | The registered study designs for predictions 1 and 2: the factual draw and sample, prediction 2's decision rule, and the OSF registration script. See `preregistration/h2/README.md` |
| `harness/` | The instrument for predictions 1 and 2: runs any memory system that speaks its protocol on the LoCoMo-Plus cognitive track and LoCoMo's factual questions, answers, judges and scores. See `harness/README.md` |

## Sources and licences

- **This repository.** The code is under the MIT licence (`LICENSE`). The second judge's verdicts and the derived tables are under CC BY 4.0 (`LICENSE-DATA`).
- **Not redistributed here.** Three sources supply the files the scripts read:
  - the whitepaper's [companion repository](https://github.com/harshappliedai/Locomo-Plus-Experiment), which declares no licence;
  - the [LoCoMo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) benchmark, which declares no licence;
  - [LoCoMo](https://github.com/snap-research/locomo), whose conversations the sample pool contains, under CC BY-NC 4.0.

  `fetch_data.sh` downloads them from the companion repository at commit `1a384b8` and checks their checksums.
- **Quoted fragments.** The judges' reasons in `data/second_judge_verdicts.jsonl` and `results/p3_verdicts.jsonl` quote short fragments of the dialogues and responses they assess. The annotators' remarks in `results/p4/` quote a word or two.

## Citation

`CITATION.cff` carries the metadata. For the paper:

```bibtex
@misc{kafton2026twodecisions,
  author       = {Kafton, Jon},
  title        = {Two Decisions, Not Three Architectures: A Reanalysis of Cognitive Memory in {LLM} Agents},
  year         = {2026},
  month        = sep,
  howpublished = {1Presence Build Log},
  url          = {https://1presence.com/blog/two-decisions-not-three-architectures}
}
```

Please also cite the work reanalysed: H. Gurawaliya, M. Ma, M. Nalenz, *Agentic Memory Realized* (appliedAI Initiative, 2026), and Y. Li et al., *Locomo-Plus* (arXiv:2602.10715).
