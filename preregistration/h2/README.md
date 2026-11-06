# Predictions 1 and 2 (H1, H2): the study designs

H2 of the registration at https://osf.io/xufd9: "Improving query planning raises factual accuracy more than cognitive accuracy." The registration fixed the prediction and its decision rule; the study design (the backbone, the two levels of planning, the speaker mapping, the item sets, the stitching, the judges per category, the exclusion rule) is registered separately before any item runs. The draft of that registration is in the 1Presence vault (`vault/Marketing/Two Decisions Not Three Architectures — Research Paper Review Sources/OSF Study Registration — H1 and H2 Study Designs — Draft 2026-11.md`); this folder holds the scripts and the drawn sample it attaches.

The system under test is 1Presence's memory pipeline, driven by `tests/eval/locomo/locomo-driver.ts` in the PresenceAI repository, which reads this repository's fetched files (`pool.json` and `locomo10.json`, both hash-checked by `fetch_data.sh`) and runs the published judge from this repository's `data/source/judge.py`. Nothing from LoCoMo or LoCoMo-Plus enters a palace, an answer call or a judge call before the registration exists: the driver refuses benchmark data without `--registration <osf id>`.

| File | What it is |
|---|---|
| `p2_draw.py` | The factual draw: LoCoMo questions stratified by category (25 per category, seed 20261104), by identifier. Run before registration; its output is `p2_factual_sample.json` |
| `p2_factual_sample.json` | The drawn questions, as identifiers and question hashes only |
| `p2_score.py` | H2's decision rule applied to a driver run directory: both-pass accuracy per track per arm, the factual gain minus the cognitive gain, a bootstrap interval, the exclusions |
| `osf_register_h2.py` | The submission: reads the vault draft by section, maps it onto the OSF Preregistration v4 template, uploads the eleven attached files, creates the draft for review, and registers it on a separate `submit`. `dry-run` makes no call |
| `osf_blocks_v4.json` | The template's question blocks, fetched from the OSF API on 2026-11-03, which the script reads the select options from |

## H1 (added 2026-11-05)

H1 of the same registration, "A bounded, curated standing state added to a retrieval pipeline raises goal and state accuracy more than causal and value accuracy", runs on the same palaces in the same driver run: two more recall arms (`h1-on`, `h1-off`: self-view off, standing state on against off), far and paired negative controls assigned by seed over the whole pool, near variants for a seeded 100-item subset, and a use judge (acted / mentioned / neither) on every reply both judges pass. Its files live in PresenceAI, `tests/eval/locomo/`: `h1-near-draw.ts` and `h1-near-subset.json` (the seeded subset), `h1-near-variants.json` (the hand-checked near variants), `h1-far-controls.json` (the far list), `h1-score.ts` (the registered rule); the answer rule and the use-judge prompt are in `tests/eval/memory-harness.ts`. `osf_register_h2.py` attaches them with H2's files.

## Order of work

1. `./fetch_data.sh` (now also fetches `locomo10.json` at a pinned commit).
2. `python3 preregistration/h2/p2_draw.py` before registration; attach `p2_factual_sample.json`.
3. Register the study design on OSF: `OSF_TOKEN=… python3 preregistration/h2/osf_register_h2.py dry-run`, then `create` (prints the review URL), review the draft at osf.io, then `submit <draft id>`. Nothing runs before that. On a Mac whose `python3` is the python.org build with no CA bundle, `urllib` fails the TLS handshake (`CERTIFICATE_VERIFY_FAILED`); prefix the command with `SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')`, or run `Install Certificates.command` once. Draft of 2026-11-05: `6aecf9b9216b1e32b96a0f55`.
4. In PresenceAI: `npx tsx tests/eval/locomo/locomo-driver.ts --registration <osf id> --factual-sample ~/two-decisions-reanalysis/preregistration/h2/p2_factual_sample.json --near-variants tests/eval/locomo/h1-near-variants.json`.
5. `python3 preregistration/h2/p2_score.py --run <run dir>` for H2 and `npx tsx tests/eval/locomo/h1-score.ts --run <run dir>` for H1.
