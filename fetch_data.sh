#!/usr/bin/env bash
# Fetch the files this reanalysis reads from the whitepaper's companion
# repository, pinned to the commit the analysis used, and check each against
# the checksum recorded in data/SOURCE_SHA256.
#
# The files are not copied into this repository: the companion repository and
# the LoCoMo-Plus data it builds on carry no licence, and the LoCoMo
# conversations inside the sample pool are CC BY-NC 4.0. They are downloaded
# from their source instead, into data/source/ (git-ignored).
set -euo pipefail

SHA=1a384b8884f487bd45da62bea5b16d1415b78403
BASE="https://raw.githubusercontent.com/harshappliedai/Locomo-Plus-Experiment/$SHA"
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/data/source"
mkdir -p "$DEST"

fetch() {  # <path in the companion repository> <local name>
  if [ -s "$DEST/$2" ]; then
    echo "  have  $2"
  else
    curl -sSL --fail -o "$DEST/$2.part" "$BASE/$1"
    mv "$DEST/$2.part" "$DEST/$2"
    echo "  got   $2"
  fi
}

echo "Fetching from harshappliedai/Locomo-Plus-Experiment @ ${SHA:0:7} into data/source/"
fetch all_three_benchmarks_cognitive_results/judge_results.json judge_results.json
fetch all_three_benchmarks_cognitive_results/responses.json     responses.json
fetch all_three_benchmarks_cognitive_results/run_meta.json      run_meta.json
fetch data/cognitive_samples_401_seed42.json                    pool.json
fetch evaluation/judge_prompts.py                               judge_prompts.py
fetch evaluation/metrics.py                                     metrics.py
fetch scripts/prepare_data.py                                   prepare_data.py
# The published judge's own code, for the registered test of prediction 3
fetch evaluation/judge.py                                       judge.py
fetch config.py                                                 config.py

echo "Checking sha256"
cd "$DEST"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c "$ROOT/data/SOURCE_SHA256"
else
  sha256sum -c "$ROOT/data/SOURCE_SHA256"
fi
