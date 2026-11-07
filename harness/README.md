# The harness: the registered design, run against any memory system

The study designs for predictions 1 and 2 of the registration at https://osf.io/xufd9 (H1: "A bounded, curated standing state added to a retrieval pipeline raises goal and state accuracy more than causal and value accuracy"; H2: "Improving query planning raises factual accuracy more than cognitive accuracy") are registered separately, before any item runs. This folder is the instrument that registration fixes: the item sets, the arms, the controls, the answer step, the judges and the scoring. It takes any memory system through one protocol, so the design can be run on a system other than the one it was written for. The system the registered study runs on is 1Presence's memory pipeline; its implementation of the protocol lives beside its product code and is attached to the registration, not published here.

## What is here

| File | What it is |
|---|---|
| `system.ts` | The protocol between the harness and a system under test, and the client and server helpers for it |
| `systems/none.ts` | The reference system: no memory. Ingest is kept, recall returns nothing. The harness runs end to end against it anywhere |
| `locomo-adapter.ts` | Reads LoCoMo and the LoCoMo-Plus pool and reproduces the companion repository's stitch; the two speaker mappings; the cognitive items and the factual questions |
| `locomo-driver.ts` | The run: base ingest with snapshots, per item the cue and the suffix, recall per arm, the answers, the judges, a summary |
| `judges.ts`, `published_judge.py` | The published judge (the companion repository's own `Judge` class, run from its hash-checked files as a Python subprocess) and the second judge (the benchmark's locked prompt through the Claude Code CLI) |
| `prompts.ts` | The texts the registration fixes: the locked cognitive judge prompt, the answer rule, the use judge's prompt; the CLI judge |
| `h1-near-draw.ts`, `h1-near-subset.json` | The seeded 100-item subset for H1's near controls |
| `h1-near-variants.json` | The hand-checked near variants for that subset |
| `h1-far-controls.json` | The far-control list |
| `h1-score.ts` | H1's registered decision rule over a run directory (H2's is `../preregistration/h2/p2_score.py`) |
| `rates.json` | Dollars per million tokens for the cost figure in `summary.json`; nothing else reads it |
| `deposit.ts` | Builds a run's release set: benchmark text replaced by SHA-256 and length, the pipeline's own outputs in full, a `SHA256SUMS` and the set's digest for timestamping |
| `fixtures/` | A synthetic two-speaker fixture in the benchmarks' exact shapes, for the smoke test; no benchmark text |

## The protocol

A system is any program that reads one JSON object per line on stdin and answers one per line on stdout, in order. `system.ts` states the shapes. The requests, in the order a run makes them:

- `hello` → the system's name, code version and models, for the run's record.
- `open` a state directory, optionally made first as a copy of a snapshot.
- `ingest` one session (date, exchanges) through the system's own save path, with the session's date as "today". Each session is its own conversation.
- `recall` what the system would hand its answer model for a text as the first message of a new conversation, with the three switches (`planning`, `self`, `standing`) at the levels given. Nothing is written. A failed or timed-out lookup is reported as a fault, and the harness excludes the item rather than scoring a miss.
- `close` the state, so its directory is complete and copyable; `snapshot` copies one.
- `exit`.

A system without one of the switches says so in `hello` and ignores that level. The harness's answer step is the same for every system: one model (`--answer-model`, default `claude-haiku-5-5`), a minimal system prompt with the date and the answer rule, the question, then what `recall` returned, as the system shapes it.

## Running it

```
npm install                       # in this folder
npm run typecheck
npm run smoke                     # the fixture, the reference system, no judges; needs ANTHROPIC_API_KEY for the answer step
```

The registered run, after the study registration exists and `../fetch_data.sh` has fetched the benchmark files:

```
npx tsx locomo-driver.ts --registration <osf id> --system "<command that runs your system>" \
    --factual-sample ../preregistration/h2/p2_factual_sample.json --near-variants h1-near-variants.json
python3 ../preregistration/h2/p2_score.py --run <run dir>     # H2
npx tsx h1-score.ts --run <run dir>                            # H1
```

The driver refuses benchmark data without `--registration`; `--fixture fixtures/synthetic` is the only way round it. The published judge needs a Python with the companion repository's two packages (`judges.ts` prints the one-line setup when it is missing) and `GEMINI_API_KEY`; the second judge and the use judge need the Claude Code CLI on a subscription. `--no-judge` runs everything up to the replies.

## What a run leaves

`meta.json` (the design, the data hashes, the harness's and the system's code versions, the models), `items.json`, `base-ingest.json`, `ingest/`, `item-prep.json`, `recalls.jsonl` (every lookup per arm, the exclusion rule's reasons), `replies.jsonl`, `verdicts.jsonl` (both judges per reply, and the use judge), `summary.json` (point estimates; the registered rules are the two scorers), and `states/` (the snapshots, which are copies and can be deleted after scoring).
