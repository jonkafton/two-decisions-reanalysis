#!/usr/bin/env python3
"""The published judge, run from the companion repository's own files.

The one sanctioned Python in this tree (root CLAUDE.md, Language): the appliedAI
companion repository's `Judge` class is Python, and OSF registration xufd9 says the
published judge runs "from its files at the commit above", hash-checked, not from a
copy. So this script loads `config.py`, `judge.py` and `judge_prompts.py` from the
reanalysis repository's `data/source/` (fetched by its `fetch_data.sh`), checks each
against `data/SOURCE_SHA256`, and serves verdicts over stdin/stdout to judges.ts.

Settings are the repository's own defaults: provider google, model gemini-2.5-flash,
temperature 0, maxOutputTokens 512, its parser. The Gemini key is read from the
environment (`GOOGLE_API_KEY`, which judges.ts fills from the project's
`GEMINI_API_KEY`) and is never printed. The OpenAI client the class constructs is
never called; its key is a placeholder.

Protocol, one JSON object per line:
  in   {"id": "...", "kind": "cognitive", "cue_dialogue": "...", "trigger_query": "...", "model_response": "..."}
       {"id": "...", "kind": "fact"|"temporal"|"adversarial", "question": "...", "gold": "...", "pred": "...", "evidence": "..."}
  out  {"id": "...", "label": "correct"|"partial"|"wrong"|null, "reason": "...", "raw": "...", "parse": "json"|"partial-json"|"invalid",
        "flag"?: "...", "error"?: "..."}
A transient error (HTTP, network) is retried five times with backoff and then
returned as `error`, never as a label. The first line out is a ready message with
the protocol lock, so the caller can record the judge's settings.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

ATTEMPTS = 5
LOCKED_SHA = "7174feb3462a2636df0c2632da7634f78ea850e9bf5069ec2ceea607d9551ca1"
CHECKED = ("config.py", "judge.py", "judge_prompts.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hash_check(source_dir, sha_file):
    want = {}
    with open(sha_file, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                want[parts[1]] = parts[0]
    for name in CHECKED:
        path = os.path.join(source_dir, name)
        if not os.path.exists(path):
            sys.exit(f"missing {path}; run ./fetch_data.sh in the reanalysis repository first")
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if want.get(name) != got:
            sys.exit(f"{name}: sha256 {got} does not match data/SOURCE_SHA256 ({want.get(name)})")
    return {name: want[name] for name in CHECKED}


def main():
    source_dir = os.environ.get("SOURCE_DIR") or sys.exit("SOURCE_DIR is required")
    sha_file = os.environ.get("SOURCE_SHA256") or sys.exit("SOURCE_SHA256 is required")
    hashes = hash_check(source_dir, sha_file)
    # The class constructs an OpenAI client it never uses here; load_settings insists on a key.
    os.environ.setdefault("OPENAI_API_KEY", "unused-by-the-published-judge")
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY is required (judges.ts fills it from GEMINI_API_KEY)")
    config = load_module("config", os.path.join(source_dir, "config.py"))
    package = types.ModuleType("evaluation")
    package.__path__ = []
    sys.modules["evaluation"] = package
    prompts = load_module("evaluation.judge_prompts", os.path.join(source_dir, "judge_prompts.py"))
    judge_module = load_module("evaluation.judge", os.path.join(source_dir, "judge.py"))
    if hashlib.sha256(prompts.COGNITIVE_AWARENESS_PROMPT.encode()).hexdigest() != LOCKED_SHA:
        sys.exit("judge prompt drifted from the locked hash")
    settings = config.load_settings()
    lock = config.assert_protocol_compliance(settings, prompts.COGNITIVE_AWARENESS_PROMPT)
    assert (lock["judge_provider"], lock["judge_model"]) == ("google", "gemini-2.5-flash"), lock
    judge = judge_module.Judge(settings)
    templates = {
        "fact": prompts.FACT_COMMONSENSE_JUDGE_PROMPT,
        "temporal": prompts.TEMPORAL_JUDGE_PROMPT,
        "adversarial": prompts.ADVERSARIAL_JUDGE_PROMPT,
    }
    prompt_hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in templates.items()}
    prompt_hashes["cognitive"] = LOCKED_SHA
    print(json.dumps({"ready": True, "protocol_lock": lock, "file_sha256": hashes, "prompt_sha256": prompt_hashes,
                      "python": sys.version.split()[0]}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        out = {"id": req.get("id")}
        try:
            if req["kind"] == "cognitive":
                prompt = prompts.COGNITIVE_AWARENESS_PROMPT.format(
                    cue_dialogue=req["cue_dialogue"], trigger_query=req["trigger_query"], model_response=req["model_response"])
                labels = ("correct", "wrong")
            else:
                tmpl = templates[req["kind"]]
                fields = {"question": req["question"], "gold": req["gold"], "pred": req["pred"]}
                if "{evidence}" in tmpl:
                    fields["evidence"] = req.get("evidence", "")
                prompt = tmpl.format(**fields)
                labels = ("correct", "partial", "wrong") if req["kind"] == "fact" else ("correct", "wrong")
            raw = None
            for attempt in range(ATTEMPTS):
                try:
                    raw = judge._run_judge_call(prompt)
                    break
                except Exception as e:  # HTTP, network, rate limit: transient
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    time.sleep(min(60, 5 * 2 ** attempt))
            if raw is None:
                out.update({"label": None, "error": err})
            else:
                # The repository's own parser. Its last resort labels unparseable output
                # "wrong" with an "Invalid JSON" reason; that is recorded as parse "invalid"
                # and flagged, as registration xufd9's H3 rule records it.
                parsed = judge_module.Judge._safe_parse(raw)
                label = str(parsed.get("label", "")).strip().lower()
                reason = str(parsed.get("reason", ""))
                parse = "invalid" if reason.startswith("Invalid JSON from judge") else "partial-json" if reason.startswith("Partial judge JSON") else "json"
                if label not in labels:
                    out.update({"label": "wrong", "reason": reason, "raw": raw, "parse": parse,
                                "flag": f"label {label!r} not in {list(labels)}; recorded wrong"})
                else:
                    out.update({"label": label, "reason": reason, "raw": raw, "parse": parse})
                    if parse == "invalid":
                        out["flag"] = "unparseable output labelled wrong by the repository's parser"
        except Exception as e:
            out.update({"label": None, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
