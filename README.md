# two-decisions-reanalysis

Code and data for **Two Decisions, Not Three Architectures: A Reanalysis of Cognitive Memory in LLM Agents** (Jon Kafton, 2026).

- Reading version: https://1presence.com/blog/two-decisions-not-three-architectures
- PDF, the version of record: https://1presence.com/papers/two-decisions-not-three-architectures-v1.pdf

The paper reanalyses the item-level results that the appliedAI Initiative released with its 2026 whitepaper *Agentic Memory Realized*. The whitepaper is a matched comparison of three agent-memory systems (Mem0, Letta and Zep) on the cognitive track of the LoCoMo-Plus benchmark. The paper also re-scores every matched response with a second judge, from a model family different from both the published judge and the answering backbone.

This repository holds the second judge's verdicts, and every script needed to reproduce the figures in the paper's §3 and Appendices A–C from them and from the whitepaper's released files.

## Findings reproduced here

- **A protocol deviation.** For 33 of the 401 item identifiers, systems were scored on different dialogues under the same identifier. 368 items remain matched.
- **The published figures, under the published judge.** On the matched items: 18.2%, 24.7% and 22.3% for Mem0, Letta and Zep; Cochran's Q = 7.11; a union of 40.8%.
- **The second judge.** It runs the same prompt on the same responses and agrees with the published judge only slightly (Cohen's κ = 0.18).
  - Accuracy is 4.3%, 2.4% and 2.2%, with no significant difference (Q = 3.68, p = 0.16).
  - The union is 8.4%.
  - The judges part on replies that are consistent with the cue but could have been written from the trigger alone.
- **The time-gap buckets are mislabelled,** and the distance curve depends on the judge.
- **The self-managing system answered with a prompt the size of the whole dialogue.** This comes from the usage records in its released responses: median about 21,100 tokens against about 21,200.

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
| `scripts/common.py` | Paths, loaders and small statistics shared by the scripts |
| `preregistration/` | The registered tests of predictions 3 and 4: runners, decision rules, the drawn sample and the annotators' instructions. See `preregistration/README.md` |

## Sources and licences

- **This repository.** The code is under the MIT licence (`LICENSE`). The second judge's verdicts and the derived tables are under CC BY 4.0 (`LICENSE-DATA`).
- **Not redistributed here.** Three sources supply the files the scripts read:
  - the whitepaper's [companion repository](https://github.com/harshappliedai/Locomo-Plus-Experiment), which declares no licence;
  - the [LoCoMo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) benchmark, which declares no licence;
  - [LoCoMo](https://github.com/snap-research/locomo), whose conversations the sample pool contains, under CC BY-NC 4.0.

  `fetch_data.sh` downloads them from the companion repository at commit `1a384b8` and checks their checksums.
- **Quoted fragments.** The judge's reasons in `data/second_judge_verdicts.jsonl` quote short fragments of the dialogues and responses they assess.

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
