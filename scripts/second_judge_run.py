#!/usr/bin/env python3
"""Second-judge pass over the released LoCoMo-Plus responses (appliedAI repo).

Re-judges every (item, system) row of the 368 matched items with the SAME locked
prompt (sha256 7174feb3…), through the Claude Code CLI (`claude --print`),
isolated from any local configuration: no tools, no MCP servers, no user or
local settings, an empty working directory, an empty system prompt and no
session persistence. The released verdicts were produced this way with
claude-opus-5-5 (CLI alias `opus`, effort medium) on 25 September 2026.

A seeded 150-row subset is judged a second time (pass 2) to measure the second
judge's own test-retest agreement, since the CLI cannot pin temperature to 0.

Resumable: results append to results/rerun_verdicts.jsonl (or $VERDICTS_OUT);
rows already present are skipped.
"""
import argparse, hashlib, importlib.util, json, os, random, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
LPX = os.environ.get('SOURCE_DIR', os.path.join(os.path.dirname(HERE), 'data', 'source'))
NEUTRAL = os.environ.get('NEUTRAL_DIR') or __import__('tempfile').mkdtemp(prefix='judge2-neutral-')
OUT = os.environ.get('VERDICTS_OUT', os.path.join(os.path.dirname(HERE), 'results', 'rerun_verdicts.jsonl'))  # never overwrites data/
LOCKED_SHA = '7174feb3462a2636df0c2632da7634f78ea850e9bf5069ec2ceea607d9551ca1'
SYSTEMS = ('mem0', 'letta', 'zep')
RATE_RE = re.compile(r'\b429\b|rate[\s-]?limit|usage limit|too many requests|limit reached|overloaded', re.I)

spec = importlib.util.spec_from_file_location('jp', os.path.join(LPX, 'judge_prompts.py'))
jp = importlib.util.module_from_spec(spec); spec.loader.exec_module(jp)
PROMPT = jp.COGNITIVE_AWARENESS_PROMPT
assert hashlib.sha256(PROMPT.encode()).hexdigest() == LOCKED_SHA, 'judge prompt drifted from the locked hash'


def safe_parse(raw: str):
    """The original judge's parser, minus its default-to-wrong fallback."""
    text = raw.strip()
    if text.startswith('```'):
        text = '\n'.join(l for l in text.splitlines() if not l.strip().startswith('```')).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and str(obj.get('label', '')).strip().lower() in ('correct', 'wrong'):
            return str(obj['label']).strip().lower(), str(obj.get('reason', ''))
    except json.JSONDecodeError:
        pass
    m = re.search(r'"label"\s*:\s*"(correct|wrong)"', text, flags=re.I)
    if m:
        return m.group(1).lower(), 'partial-json'
    return None, None


def build_rows(retest_n: int, seed: int):
    pool = {p['sample_id']: p for p in json.load(open(os.path.join(LPX, 'pool.json')))}
    resp = {(r['sample_id'], r['system']): r for r in json.load(open(os.path.join(LPX, 'responses.json')))}
    matched = json.load(open(os.path.join(os.path.dirname(HERE), 'data', 'matched_items.json')))
    rows = []
    for sid in matched:
        for sysname in SYSTEMS:
            r = resp.get((sid, sysname))
            if r is None:
                raise SystemExit(f'missing response {sid}/{sysname}')
            p = pool[sid]
            rows.append({'sample_id': sid, 'system': sysname, 'pass': 1,
                         'prompt': PROMPT.format(cue_dialogue=p['cue_dialogue'],
                                                 trigger_query=p['trigger_query'],
                                                 model_response=r['response'])})
    rng = random.Random(seed)
    rng.shuffle(rows)
    retest = [dict(r, **{'pass': 2}) for r in rng.sample(rows, retest_n)]
    return rows + retest


def call(row, model: str, effort: str, timeout: int):
    args = ['claude', '--print', '--output-format', 'json', '--model', model, '--effort', effort,
            '--tools', '', '--strict-mcp-config', '--setting-sources', 'project',
            '--no-session-persistence', '--system-prompt', '']
    t0 = time.time()
    try:
        proc = subprocess.run(args, input=row['prompt'], capture_output=True, text=True,
                              cwd=NEUTRAL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'error': 'timeout', 'wall_s': round(time.time() - t0, 2)}
    out = proc.stdout.strip()
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        blob = (out + ' ' + proc.stderr)[-600:]
        return {'error': 'rate_limited' if RATE_RE.search(blob) else 'bad_output', 'detail': blob,
                'wall_s': round(time.time() - t0, 2)}
    if d.get('is_error'):
        blob = json.dumps(d)[-600:]
        return {'error': 'rate_limited' if RATE_RE.search(blob) else 'cli_error', 'detail': blob,
                'wall_s': round(time.time() - t0, 2)}
    raw = d.get('result') or ''
    label, reason = safe_parse(raw)
    u = d.get('usage') or {}
    return {'label': label, 'reason': reason, 'raw': raw,
            'model': ','.join((d.get('modelUsage') or {}).keys()),
            'cost_usd_list': d.get('total_cost_usd'),
            'in': u.get('input_tokens'), 'cache_w': u.get('cache_creation_input_tokens'),
            'cache_r': u.get('cache_read_input_tokens'), 'out': u.get('output_tokens'),
            'think': (u.get('output_tokens_details') or {}).get('thinking_tokens'),
            'api_ms': d.get('duration_api_ms'), 'wall_s': round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='opus')
    ap.add_argument('--effort', default='medium')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0, help='only the first N pending rows (calibration)')
    ap.add_argument('--retest', type=int, default=150)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--timeout', type=int, default=240)
    a = ap.parse_args()

    rows = build_rows(a.retest, a.seed)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get('label') in ('correct', 'wrong'):
                done.add((j['sample_id'], j['system'], j['pass']))
    pending = [r for r in rows if (r['sample_id'], r['system'], r['pass']) not in done]
    if a.limit:
        pending = pending[:a.limit]
    print(f'total rows {len(rows)} | done {len(done)} | running {len(pending)} | model {a.model} effort {a.effort}', flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    stats = {'ok': 0, 'fail': 0, 'rate': 0}

    def work(row):
        for attempt in range(4):
            if stop.is_set():
                return
            res = call(row, a.model, a.effort, a.timeout)
            if res.get('error') == 'rate_limited':
                with lock:
                    stats['rate'] += 1
                    if stats['rate'] >= 6:
                        stop.set()  # usage limit reached: stop cleanly; rerun to resume
                        print('RATE LIMITED repeatedly — stopping; rerun to resume', flush=True)
                        return
                time.sleep(90 * (attempt + 1))
                continue
            if res.get('label') in ('correct', 'wrong'):
                break
            time.sleep(5)
        rec = {'sample_id': row['sample_id'], 'system': row['system'], 'pass': row['pass'],
               'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), **res}
        with lock:
            with open(OUT, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            if res.get('label') in ('correct', 'wrong'):
                stats['ok'] += 1
            else:
                stats['fail'] += 1
            n = stats['ok'] + stats['fail']
            if n % 50 == 0 or n == len(pending):
                print(f'{time.strftime("%H:%M:%S")} {n}/{len(pending)} ok={stats["ok"]} fail={stats["fail"]} rate_hits={stats["rate"]}', flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(as_completed([ex.submit(work, r) for r in pending]))
    print(f'finished ok={stats["ok"]} fail={stats["fail"]} rate_hits={stats["rate"]} stopped={stop.is_set()}', flush=True)


if __name__ == '__main__':
    main()
