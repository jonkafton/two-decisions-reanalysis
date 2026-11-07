# Deposit: the registered run of the H1 and H2 study designs (OSF registration xf3gk)

This is the per-item record of the run registered at https://osf.io/xf3gk, made on 2026-11-06 and 2026-11-07 with the public harness at https://github.com/jonkafton/two-decisions-reanalysis (`harness/`, commit `82fed76`) against 1Presence's memory pipeline, and scored with the two registered scorers. It is built by `harness/deposit.ts` from the run directories.

## What is here

| Path | Contents |
|---|---|
| `shard-1/` … `shard-4/` | One directory per shard of the run (the items were split four ways, `--shard k/4`): `meta.json` (the design, data hashes, code versions, models), `items.json`, `base-ingest.json`, `ingest/<base>.jsonl` (the save pipeline's record per session), `item-prep.json`, `recalls.jsonl` (every lookup per item and arm), `replies.jsonl`, `verdicts.jsonl` (both judges per reply and the use judge), `summary.json` (the driver's point estimates) |
| `merged/` | The four shards' `items.json`, `recalls.jsonl`, `replies.jsonl` and `verdicts.jsonl` concatenated, with the merge recorded in `meta.json`; `p2_result.json` is the registered H2 scorer's output on it |
| `h1_result.json` | The registered H1 scorer's output over the four shards |
| `SHA256SUMS`, `deposit.sha256` | A digest of every file, and the digest of that list; `deposit.sha256.ots` is its OpenTimestamps proof when present |

## The benchmarks' text is not here

LoCoMo is CC BY-NC 4.0 and LoCoMo-Plus carries no licence, so no benchmark text is redistributed. Wherever a run record held benchmark text, this deposit holds instead two fields, `<name>Sha256` (the SHA-256 of the text as UTF-8, exactly as the record held it) and `<name>Chars` (its length). A reader with the benchmark files can restore any such text by hashing the candidate string: every cue dialogue, trigger query, trigger text and cue turn of the 401 cognitive items hashes to a string that appears verbatim in the LoCoMo-Plus pool file, and every question, answer and evidence turn of the 125 factual questions to a string in `locomo10.json`. The files and their digests are in the repository's `data/SOURCE_SHA256` and are fetched by `fetch_data.sh`. The one exception is the `gold` of the 25 adversarial questions, which is the harness's own fixed sentence rather than the benchmark's, and is kept in plain.

Hashed: the item texts above; the probe text of every recall (`text`: for the `trigger` and `paired` probes a benchmark trigger; for `near` and `far` probes the harness's own texts, published in `harness/h1-near-variants.json` and `harness/h1-far-controls.json`); the text of every lookup result the answer model saw (`seen`, and each `tool_result` block in `messages`); the facts the capture judge extracted and the text of every memory write in the ingest records.

Kept in full: the lookups' inputs (the planner's queries, in `lookups[].input` and the `tool_use` blocks), the standing block (`standingSeen`) and the self-view (`selfSeen`) as read at each recall, every reply (`replies.jsonl`), every verdict with the judge's raw output (`verdicts.jsonl`), every usage figure, count and timing, and the scorers' outputs. These are the pipeline's own writing. A reply or a standing block may quote the conversation it answers; that is the pipeline's output as judged, and is deposited as registered.

Dropped: the tool schemas (the product's), the snapshot paths, and the `states/` directories (copies of the memory store).

## Reading the run

`recalls.jsonl`: one row per (track, item, arm, probe). `arm` is `on`/`off` for H2 (the planner on or off) and `h1-on`/`h1-off` for H1 (the standing block on or off); `probe` is `trigger`, or a control (`far`, `paired`, `near`). `excluded` and `reason` carry the registered exclusion rule's outcome (none was excluded in this run). `replies.jsonl` rows carry the H2 arms as `system-on`, `system-off` and `none` (no memory). `verdicts.jsonl`: `judge` is `published` (the companion repository's own judge class and prompt, run from its hash-checked files), `second` (the benchmark's locked prompt through a second model family) or `use` (the use judge on replies both judges passed); `parse` records whether the published judge's output was clean JSON, partial JSON recovered by that repository's own parser, or invalid and labelled wrong by it.
