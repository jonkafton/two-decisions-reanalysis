#!/usr/bin/env python3
"""Prediction 3: a model given only the trigger, scored by both judges.

Registered on OSF before any real call was made. Until that registration is
approved, run this only with --stub: every real call on these items observes
the outcome that prediction 3 is about.

  Backbone      The whitepaper's own settings: gpt-4o-mini, temperature 0,
                max_tokens 512. The trigger, exactly as the systems received it,
                is the only user message. The system message is the first
                sentence of the systems' own answer prompt with its memory
                section removed: "You are a helpful assistant."
  Published     The companion repository's own Judge class and settings, loaded
  judge         from the files ./fetch_data.sh downloads at the pinned commit and
                hash-checks: gemini-2.5-flash through the Generative Language
                API, temperature 0, maxOutputTokens 512, the locked prompt, and
                that repository's own parser.
  Second judge  claude-opus-5-5 through the Claude Code CLI, invoked exactly as
                scripts/second_judge_run.py invokes it, with the full model id.

An empty reply is scored wrong by both judges without a call. Transient errors
(network, HTTP, rate limits) are retried and never recorded as verdicts; a run
that still fails stops and resumes where it left off. A judge reply that cannot
be parsed after five attempts is recorded as wrong and flagged. Outputs append
to results/p3_replies.jsonl and results/p3_verdicts.jsonl; p3_score.py applies
the registered decision rule.

  Real run (after approval):
    OPENAI_API_KEY=... GOOGLE_API_KEY=... python3 preregistration/p3_trigger_only.py
  Stub run (no network, no keys):
    python3 preregistration/p3_trigger_only.py --stub --out <scratch dir>

The real run needs `pip install -r preregistration/requirements-p3.txt` and the
Claude Code CLI; the stub run needs neither.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import RESULTS, SOURCE, matched_items, pool  # noqa: E402

LOCKED_SHA = "7174feb3462a2636df0c2632da7634f78ea850e9bf5069ec2ceea607d9551ca1"
BACKBONE = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."
SECOND_JUDGE_MODEL = "claude-opus-5-5"
SECOND_JUDGE_EFFORT = "medium"
ATTEMPTS = 5
LOCK = threading.Lock()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_stub_sdks():
    """Stand-ins for the two packages the companion repository imports, so a
    stub run exercises its code with no network and no installs."""
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, **kwargs):
            pass

    openai.OpenAI = OpenAI
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["openai"] = openai
    sys.modules["dotenv"] = dotenv
    os.environ["OPENAI_API_KEY"] = "stub-no-network"
    os.environ["GOOGLE_API_KEY"] = "stub-no-network"


def published_judge(stub):
    for name in ("config.py", "judge.py", "judge_prompts.py"):
        if not os.path.exists(os.path.join(SOURCE, name)):
            sys.exit(f"missing {name} in {SOURCE}; run ./fetch_data.sh first")
    config = load_module("config", os.path.join(SOURCE, "config.py"))
    package = types.ModuleType("evaluation")
    package.__path__ = []
    sys.modules["evaluation"] = package
    prompts = load_module("evaluation.judge_prompts", os.path.join(SOURCE, "judge_prompts.py"))
    judge_module = load_module("evaluation.judge", os.path.join(SOURCE, "judge.py"))
    prompt = prompts.COGNITIVE_AWARENESS_PROMPT
    assert hashlib.sha256(prompt.encode()).hexdigest() == LOCKED_SHA, "judge prompt drifted"
    settings = config.load_settings()
    lock = config.assert_protocol_compliance(settings, prompt)
    assert (lock["judge_provider"], lock["judge_model"], lock["backbone_model"]) == \
        ("google", "gemini-2.5-flash", BACKBONE), lock
    judge = judge_module.Judge(settings)
    if stub:
        def canned(p, _n=[0]):
            _n[0] += 1
            if _n[0] % 9 == 0:
                return '{"label": "correct", "reason": "stub"}'
            if _n[0] % 13 == 0:
                return '{"label": "correct", "reas'  # exercises the partial-JSON path
            return '{"label": "wrong", "reason": "stub"}'
        judge._run_judge_call = canned
    return judge, lock


def second_judge(stub):
    module = load_module("second_judge_run", os.path.join(ROOT, "scripts", "second_judge_run.py"))
    if stub:
        def call(prompt, _n=[0]):
            _n[0] += 1
            return {"label": "correct" if _n[0] % 40 == 0 else "wrong", "raw": "stub", "model": "stub"}
        return call, module.PROMPT

    def call(prompt):
        for attempt in range(ATTEMPTS):
            res = module.call({"prompt": prompt}, SECOND_JUDGE_MODEL, SECOND_JUDGE_EFFORT, 240)
            if res.get("error") in ("rate_limited", "timeout", "cli_error"):
                time.sleep(90 * (attempt + 1))
                continue
            if res.get("label") in ("correct", "wrong"):
                return res
            time.sleep(5)
        if res.get("error") in ("rate_limited", "timeout", "cli_error"):
            raise RuntimeError(f"second judge unavailable: {res.get('error')}")
        return dict(res, label="wrong", flag="unparsed after retries")
    return call, module.PROMPT


def backbone(stub):
    if stub:
        def answer(trigger, _n=[0]):
            _n[0] += 1
            text = "" if _n[0] % 50 == 0 else f"[stub reply {_n[0]}] That sounds like a lot to think about."
            return {"reply": text, "model": "stub", "finish_reason": "stop"}
        return answer
    from openai import OpenAI
    kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**kwargs)

    def answer(trigger):
        c = client.chat.completions.create(
            model=BACKBONE, temperature=0, max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": trigger}])
        u = c.usage
        return {"reply": c.choices[0].message.content or "", "model": c.model,
                "system_fingerprint": getattr(c, "system_fingerprint", None),
                "finish_reason": c.choices[0].finish_reason,
                "tokens_in": getattr(u, "prompt_tokens", None), "tokens_out": getattr(u, "completion_tokens", None)}
    return answer


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def append(path, row):
    with LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def with_retries(fn, what):
    for attempt in range(ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # transient: network, HTTP, rate limits
            last = e
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"{what} failed after {ATTEMPTS} attempts: {last}")


def run(items, workers, fn):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, sid) for sid in items]
        for n, fut in enumerate(as_completed(futures), start=1):
            fut.result()
            if n % 50 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="no network: stand-in models, for testing the plumbing")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.stub:
        install_stub_sdks()
    os.makedirs(args.out, exist_ok=True)
    replies_path = os.path.join(args.out, "p3_replies.jsonl")
    verdicts_path = os.path.join(args.out, "p3_verdicts.jsonl")

    items, p = matched_items(), pool()
    assert len(items) == 368
    judge, lock = published_judge(args.stub)
    second, prompt = second_judge(args.stub)
    answer = backbone(args.stub)
    print(f"protocol lock: {json.dumps(lock)}")
    print(f"{'STUB RUN: no network, stand-in models' if args.stub else 'REAL RUN'}; writing to {args.out}")
    meta_path = os.path.join(args.out, "p3_meta.jsonl")
    cli = "stub" if args.stub else __import__("subprocess").run(
        ["claude", "--version"], capture_output=True, text=True).stdout.strip()
    append(meta_path, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stub": args.stub, "protocol_lock": lock,
                       "backbone": BACKBONE, "backbone_temperature": 0, "backbone_max_tokens": 512,
                       "system_prompt": SYSTEM_PROMPT, "second_judge_model": SECOND_JUDGE_MODEL,
                       "second_judge_effort": SECOND_JUDGE_EFFORT, "claude_cli_version": cli})

    done = {r["sample_id"] for r in read_jsonl(replies_path)}
    print(f"stage 1, trigger-only replies: {len(done)} done, {len(items) - len(done)} to go")

    def reply(sid):
        r = with_retries(lambda: answer(p[sid]["trigger_query"]), f"backbone {sid}")
        append(replies_path, {"sample_id": sid, **r, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    run([s for s in items if s not in done], args.workers, reply)
    replies = {r["sample_id"]: r["reply"] for r in read_jsonl(replies_path)}
    assert set(replies) == set(items), "replies incomplete"

    judged = {(r["sample_id"], r["judge"]) for r in read_jsonl(verdicts_path)}

    def published(sid):
        text = replies[sid]
        if not text.strip():
            row = {"label": "wrong", "flag": "empty reply, scored without a call"}
        else:
            try:
                j = with_retries(lambda: judge.evaluate(cue_dialogue=p[sid]["cue_dialogue"],
                                                        trigger_query=p[sid]["trigger_query"],
                                                        model_response=text), f"published judge {sid}")
                row = {"label": j["label"], "reason": j["reason"], "raw": j["raw"]}
            except RuntimeError as e:
                if "ValueError" in repr(e.__cause__) or "invalid label" in str(e):
                    row = {"label": "wrong", "flag": f"unparsed: {e}"}
                else:
                    raise
        append(verdicts_path, {"sample_id": sid, "judge": "published", "model": lock["judge_model"],
                               **row, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def second_verdict(sid):
        text = replies[sid]
        if not text.strip():
            row = {"label": "wrong", "flag": "empty reply, scored without a call"}
        else:
            row = second(prompt.format(cue_dialogue=p[sid]["cue_dialogue"],
                                       trigger_query=p[sid]["trigger_query"], model_response=text))
        append(verdicts_path, {"sample_id": sid, "judge": "second", **row,
                               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    todo = [s for s in items if (s, "published") not in judged]
    print(f"stage 2, published judge: {len(todo)} to go")
    run(todo, args.workers, published)
    todo = [s for s in items if (s, "second") not in judged]
    print(f"stage 3, second judge: {len(todo)} to go")
    run(todo, args.workers, second_verdict)
    print(f"done; score with: python3 preregistration/p3_score.py --dir {args.out}")


if __name__ == "__main__":
    main()
