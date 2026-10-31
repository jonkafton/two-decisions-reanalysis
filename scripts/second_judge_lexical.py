#!/usr/bin/env python3
"""Judge-free check (paper §3 and Appendix B): does a response carry any content word from the cue that
the trigger does not already contain? A crude recall signal — paraphrase evades
it, and a shared word can be coincidence — but it needs no judge, so it can
say which judge's positives look like recall."""
import json, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LPX = os.environ.get('SOURCE_DIR', os.path.join(os.path.dirname(HERE), 'data', 'source'))
STOP = set("""a an the and or but if so of to in on at by for with from into onto about as is are was were be been being am do does did
doing have has had having i me my mine myself you your yours we our ours they them their theirs he him his she her hers it its this that these
those there here what which who whom whose when where why how all any both each few more most other some such no nor not only own same than too
very can will just should could would may might must shall now then ever never always also still even really quite rather lot lots much many
thing things something anything everything nothing someone anyone everyone time times way ways day days week weeks month months year years
feel feeling felt think thinking thought know knew make made get got getting go going went gone come came take took see saw look looking say said
tell told want wanted need needed like liked love loved good great nice sure well yeah yes okay ok oh hey hi thanks thank because while though
although since until again back out over under up down off just one two three first last next new old big small little right left""".split())


def stem(w):
    for suf in ('ingly', 'edly', 'ing', 'ied', 'ies', 'ed', 'es', 's', 'ly'):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def terms(text):
    return {stem(w) for w in re.findall(r"[a-z][a-z'\-]+", text.lower()) if len(w) >= 4 and w not in STOP}


def final_text(sysname, s):
    if sysname == 'letta':
        m = re.findall(r"AssistantMessage\(.*?content=(\"|')(.*?)(?<!\\)\1, date=", s, flags=re.S)
        if m:
            return ' '.join(x[1] for x in m)
    return s


pool = {p['sample_id']: p for p in json.load(open(os.path.join(LPX, 'pool.json')))}
matched = json.load(open(os.path.join(os.path.dirname(HERE), 'data', 'matched_items.json')))
resp = {(r['sample_id'], r['system']): final_text(r['system'], r['response']) for r in json.load(open(os.path.join(LPX, 'responses.json')))}
j1 = {(r['sample_id'], r['system']): int(r['label'] == 'correct') for r in json.load(open(os.path.join(LPX, 'judge_results.json')))}
j2 = {}
for line in open(os.path.join(os.path.dirname(HERE), 'data', 'second_judge_verdicts.jsonl')):
    r = json.loads(line)
    if r['pass'] == 1 and r.get('label') in ('correct', 'wrong'):
        j2[(r['sample_id'], r['system'])] = int(r['label'] == 'correct')

# document frequency of each stem across every released response: a cue word that
# appears in many responses ("challenge", "understand") is not evidence of recall
from collections import Counter
DF = Counter()
for v in resp.values():
    DF.update(terms(v))
NRESP = len(resp)
RARE = float(os.environ.get('RARE', '0.03'))  # the paper reports 0.02, 0.03 and 0.05

cells = defaultdict(lambda: [0, 0])
examples = defaultdict(list)
for i in matched:
    p = pool[i]
    cue_only = {t for t in terms(p['cue_dialogue']) - terms(p['trigger_query']) if DF[t] / NRESP < RARE and "'" not in t}
    for s in ('mem0', 'letta', 'zep'):
        k = (i, s)
        if k not in j2:
            continue
        hit = bool(cue_only & terms(resp[k]))
        tag = f'J1{"+" if j1[k] else "-"} J2{"+" if j2[k] else "-"}'
        cells[tag][0] += hit
        cells[tag][1] += 1
        if hit and len(examples[tag]) < 3:
            examples[tag].append((k, sorted(cue_only & terms(resp[k]))[:6]))
out = {t: {'with_cue_word': v[0], 'n': v[1], 'pct': round(100 * v[0] / v[1], 1) if v[1] else None} for t, v in sorted(cells.items())}
print(json.dumps(out, indent=1))
print(json.dumps({t: [(list(k), w) for k, w in ex] for t, ex in examples.items()}, indent=1)[:1500])
a, b, c = (out[t]['with_cue_word'] / out[t]['n'] for t in ('J1+ J2+', 'J1+ J2-', 'J1- J2-'))
print(f"rarity threshold {RARE}: share of the published judge's solo passes carrying the cue, read as a mixture: {100 * (b - c) / (a - c):.1f}%")
