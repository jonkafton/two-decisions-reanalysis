#!/usr/bin/env python3
"""Two-judge reanalysis of the LoCoMo-Plus verdicts on the 368 matched items
(paper §3 and Appendix B).

J1 = the released verdicts (the published judge). J2 = the second judge
(data/second_judge_verdicts.jsonl, pass 1); pass 2 is J2's own retest subset.
Writes results/second_judge_analysis.json and prints it. Run ./fetch_data.sh
first; needs numpy and scipy. BOOT and PBOOT set the bootstrap draws
(defaults 200 and 100, as released).

Latent-class models (Dawid-Skene style, judges conditionally independent given
the truth, judge error rates shared across the three systems):
  full        any of the 8 truth patterns over (mem0, letta, zep)
  nested      truth sets nested in some order (no complementarity), best of 6
  identical   the three systems solve exactly the same items (the v1 null)
"""
import itertools, json, math, os, random, sys
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LPX = os.environ.get('SOURCE_DIR', os.path.join(os.path.dirname(HERE), 'data', 'source'))
SYS = ('mem0', 'letta', 'zep')
NAME = {'mem0': 'extraction', 'letta': 'self-managing', 'zep': 'graph'}

matched = json.load(open(os.path.join(os.path.dirname(HERE), 'data', 'matched_items.json')))
pool = {p['sample_id']: p for p in json.load(open(os.path.join(LPX, 'pool.json')))}
j1 = {(r['sample_id'], r['system']): int(r['label'] == 'correct') for r in json.load(open(os.path.join(LPX, 'judge_results.json')))}
j2, rt = {}, {}
for line in open(os.path.join(os.path.dirname(HERE), 'data', 'second_judge_verdicts.jsonl')):
    r = json.loads(line)
    if r.get('label') not in ('correct', 'wrong'):
        continue
    (j2 if r['pass'] == 1 else rt)[(r['sample_id'], r['system'])] = int(r['label'] == 'correct')

items = [i for i in matched if all((i, s) in j2 for s in SYS)]
N = len(items)
Y = np.array([[[j1[(i, s)], j2[(i, s)]] for s in SYS] for i in items])  # N x 3 x 2
out = {'n_items': N, 'n_j2_rows': len(j2), 'n_retest_rows': len(rt)}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return round(float((po - pe) / (1 - pe)), 3), round(100 * float(po), 1)


def mcnemar_exact(x, y):
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    return b, c, p


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, run = [0] * len(ps), 0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = run
    return adj


def cochran_q(M):
    k = M.shape[1]; T = M.sum(1); C = M.sum(0); N_ = M.sum()
    num = (k - 1) * (k * (C ** 2).sum() - N_ ** 2)
    den = k * N_ - (T ** 2).sum()
    q = num / den if den else 0.0
    return round(float(q), 2), float(stats.chi2.sf(q, k - 1))


def labels(which):
    if which == 'J1': return Y[:, :, 0]
    if which == 'J2': return Y[:, :, 1]
    if which == 'both': return Y[:, :, 0] & Y[:, :, 1]
    if which == 'either': return Y[:, :, 0] | Y[:, :, 1]


# ── 1. accuracy, ranking and union under each labelling ──────────────────────
for which in ('J1', 'J2', 'both', 'either'):
    L = labels(which)
    acc = {s: wilson(int(L[:, k].sum()), N) for k, s in enumerate(SYS)}
    q, qp = cochran_q(L)
    pairs = [('letta', 'mem0'), ('zep', 'mem0'), ('letta', 'zep')]
    raw = [mcnemar_exact(L[:, SYS.index(a)], L[:, SYS.index(b)]) for a, b in pairs]
    adj = holm([r[2] for r in raw])
    union = L.max(1)
    lead = SYS[int(np.argmax(L.sum(0)))]
    not_leader = int(((union == 1) & (L[:, SYS.index(lead)] == 0)).sum())
    k_per_item = Counter(L.sum(1).tolist())
    out[which] = {
        'accuracy': acc, 'cochran_q': q, 'cochran_p': round(qp, 4),
        'pairwise': {f'{a}-{b}': {'b': r[0], 'c': r[1], 'p_raw': round(r[2], 4), 'p_holm': round(p, 4)} for (a, b), r, p in zip(pairs, raw, adj)},
        'union': wilson(int(union.sum()), N), 'leader': lead, 'union_not_leader': (not_leader, wilson(not_leader, N)),
        'systems_correct_per_item': {int(k): v for k, v in sorted(k_per_item.items())},
    }

# ── 2. agreement between the judges, and J2 against itself ───────────────────
out['agreement_J1_J2'] = {'all': kappa(Y[:, :, 0].ravel(), Y[:, :, 1].ravel())}
for k, s in enumerate(SYS):
    out['agreement_J1_J2'][s] = kappa(Y[:, k, 0], Y[:, k, 1])
tab = Counter(zip(Y[:, :, 0].ravel().tolist(), Y[:, :, 1].ravel().tolist()))
out['j1_j2_table'] = {'J1+J2+': tab[(1, 1)], 'J1+J2-': tab[(1, 0)], 'J1-J2+': tab[(0, 1)], 'J1-J2-': tab[(0, 0)]}
pairs_rt = [(j2[key], rt[key]) for key in rt if key in j2 and key[0] in set(matched)]
if pairs_rt:
    a, b = zip(*pairs_rt)
    out['j2_retest'] = {'n': len(pairs_rt), 'kappa_pct': kappa(a, b)}
    out['j1_rows_in_retest_vs_j2'] = kappa([j1[k] for k in rt if k in j2], [j2[k] for k in rt if k in j2])

# ── 3. do J1's sole-correct cells replicate under J2? ────────────────────────
rep = defaultdict(lambda: [0, 0])
J1 = Y[:, :, 0]
for i in range(N):
    n1 = int(J1[i].sum())
    for k in range(3):
        if J1[i, k] == 1:
            tag = {1: 'sole', 2: 'two', 3: 'all_three'}[n1]
            rep[tag][0] += int(Y[i, k, 1]); rep[tag][1] += 1
out['replication_of_J1_correct_by_J1_pattern'] = {t: (v[0], v[1], wilson(v[0], v[1])) for t, v in rep.items()}
rep2 = defaultdict(lambda: [0, 0])
J2 = Y[:, :, 1]
for i in range(N):
    n2 = int(J2[i].sum())
    for k in range(3):
        if J2[i, k] == 1:
            tag = {1: 'sole', 2: 'two', 3: 'all_three'}[n2]
            rep2[tag][0] += int(Y[i, k, 0]); rep2[tag][1] += 1
out['replication_of_J2_correct_by_J2_pattern'] = {t: (v[0], v[1], wilson(v[0], v[1])) for t, v in rep2.items()}

# ── 4. latent-class models ───────────────────────────────────────────────────
ALL = list(itertools.product((0, 1), repeat=3))


def nested_sets(order):
    a, b, c = order  # a strongest ... c weakest
    pats = [(0, 0, 0)]
    for upto in range(1, 4):
        t = [0, 0, 0]
        for s in order[:upto]:
            t[s] = 1
        pats.append(tuple(t))
    return pats


def fit(Yd, patterns, restarts=12, iters=3000, seed=0, init=None):
    rng = np.random.default_rng(seed)
    P = np.array(patterns)                      # T x 3
    best = None
    for r in range(restarts):
        if init is not None and r == 0:
            pi, se, sp = init['pi'].copy(), init['se'].copy(), init['sp'].copy()
        else:
            pi = rng.dirichlet(np.ones(len(P)))
            se = rng.uniform(0.6, 0.95, 2); sp = rng.uniform(0.75, 0.98, 2)
        prev = -np.inf
        for _ in range(iters):
            # log P(y | t) for each item and pattern
            ll = np.zeros((Yd.shape[0], len(P)))
            for j in range(2):
                y = Yd[:, :, j][:, None, :]          # N x 1 x 3
                t = P[None, :, :]                     # 1 x T x 3
                p1 = np.where(t == 1, se[j], 1 - sp[j])
                ll += np.where(y == 1, np.log(p1), np.log(1 - p1)).sum(2)
            lj = ll + np.log(np.clip(pi, 1e-300, None))[None, :]
            m = lj.max(1, keepdims=True)
            tot = m[:, 0] + np.log(np.exp(lj - m).sum(1))
            loglik = float(tot.sum())
            R = np.exp(lj - tot[:, None])             # N x T
            pi = R.mean(0)
            for j in range(2):
                y = Yd[:, :, j]
                w1 = R @ P                            # N x 3: P(t_s=1)
                se[j] = np.clip((w1 * y).sum() / w1.sum(), 1e-4, 1 - 1e-4)
                w0 = 1 - w1
                sp[j] = np.clip((w0 * (1 - y)).sum() / w0.sum(), 1e-4, 1 - 1e-4)
            if loglik - prev < 1e-9:
                break
            prev = loglik
        if (se + sp > 1).all() and (best is None or loglik > best['ll']):
            best = {'ll': loglik, 'pi': pi.copy(), 'se': se.copy(), 'sp': sp.copy(), 'patterns': patterns}
    return best


def derived(f):
    P = np.array(f['patterns']); pi = f['pi']
    acc = {s: float((pi * P[:, k]).sum()) for k, s in enumerate(SYS)}
    lead = max(acc, key=acc.get); li = SYS.index(lead)
    union = float(pi[P.sum(1) > 0].sum())
    unl = float(pi[(P.sum(1) > 0) & (P[:, li] == 0)].sum())
    return {'acc': acc, 'union': union, 'union_not_leader': unl, 'leader': lead,
            'se': f['se'].tolist(), 'sp': f['sp'].tolist()}


full = fit(Y, ALL, seed=1)
nested = max((fit(Y, nested_sets(o), seed=2) | {'order': o} for o in itertools.permutations(range(3))), key=lambda f: f['ll'])
ident = fit(Y, [(0, 0, 0), (1, 1, 1)], seed=3)
indep = None  # per-system independent truths: product of three Bernoullis
def fit_indep(Yd, restarts=8, iters=4000, seed=4):
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        p = rng.uniform(0.1, 0.4, 3); se = rng.uniform(0.6, 0.95, 2); sp = rng.uniform(0.75, 0.98, 2); prev = -np.inf
        for _ in range(iters):
            lp1 = np.log(p)[None, :].repeat(Yd.shape[0], 0); lp0 = np.log(1 - p)[None, :].repeat(Yd.shape[0], 0)
            for j in range(2):
                y = Yd[:, :, j]
                lp1 = lp1 + np.where(y == 1, np.log(se[j]), np.log(1 - se[j]))
                lp0 = lp0 + np.where(y == 1, np.log(1 - sp[j]), np.log(sp[j]))
            m = np.maximum(lp1, lp0); tot = m + np.log(np.exp(lp1 - m) + np.exp(lp0 - m))
            ll = float(tot.sum()); w1 = np.exp(lp1 - tot)
            p = np.clip(w1.mean(0), 1e-4, 1 - 1e-4)
            for j in range(2):
                y = Yd[:, :, j]
                se[j] = np.clip((w1 * y).sum() / w1.sum(), 1e-4, 1 - 1e-4)
                sp[j] = np.clip(((1 - w1) * (1 - y)).sum() / (1 - w1).sum(), 1e-4, 1 - 1e-4)
            if ll - prev < 1e-9: break
            prev = ll
        if (se + sp > 1).all() and (best is None or ll > best['ll']):
            best = {'ll': ll, 'p': p.tolist(), 'se': se.tolist(), 'sp': sp.tolist()}
    return best
indep = fit_indep(Y)

k_params = {'full': 7 + 4, 'nested': 3 + 4, 'identical': 1 + 4, 'independent': 3 + 4}
models = {'full': full['ll'], 'nested': nested['ll'], 'identical': ident['ll'], 'independent': indep['ll']}
out['models'] = {m: {'loglik': round(ll, 2), 'k': k_params[m], 'AIC': round(2 * k_params[m] - 2 * ll, 2), 'BIC': round(k_params[m] * math.log(N) - 2 * ll, 2)} for m, ll in models.items()}
out['models']['nested']['order_strongest_first'] = [SYS[k] for k in nested['order']]
out['LRT_full_vs_nested'] = {'stat': round(2 * (full['ll'] - nested['ll']), 2), 'df': 4, 'p': float(stats.chi2.sf(2 * (full['ll'] - nested['ll']), 4))}
out['LRT_full_vs_identical'] = {'stat': round(2 * (full['ll'] - ident['ll']), 2), 'df': 6, 'p': float(stats.chi2.sf(2 * (full['ll'] - ident['ll']), 6))}
out['LRT_full_vs_independent'] = {'stat': round(2 * (full['ll'] - indep['ll']), 2), 'df': 4, 'p': float(stats.chi2.sf(2 * (full['ll'] - indep['ll']), 4))}
out['full_fit'] = derived(full) | {'pi': {''.join(map(str, t)): round(float(p), 4) for t, p in zip(ALL, full['pi'])}}
out['nested_fit'] = derived(nested)
out['identical_fit'] = derived(ident)

# parametric bootstrap for the full-vs-nested LRT (boundary-safe), and item bootstrap CIs
B = int(os.environ.get('BOOT', '200'))
rng = random.Random(7)
boot = defaultdict(list)
for b in range(B):
    idx = [rng.randrange(N) for _ in range(N)]
    f = fit(Y[idx], ALL, restarts=3, iters=1500, seed=100 + b, init=full)
    if f is None: continue
    d = derived(f)
    boot['union'].append(d['union']); boot['union_not_leader'].append(d['union_not_leader'])
    for s in SYS: boot['acc_' + s].append(d['acc'][s])
    boot['se1'].append(d['se'][0]); boot['sp1'].append(d['sp'][0]); boot['se2'].append(d['se'][1]); boot['sp2'].append(d['sp'][1])
out['full_fit_bootstrap_95'] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in boot.items()}

PB = int(os.environ.get('PBOOT', '100'))
null_stats = []
nrng = np.random.default_rng(11)
Pn = np.array(nested['patterns'])
for b in range(PB):
    t_idx = nrng.choice(len(Pn), size=N, p=nested['pi'] / nested['pi'].sum())
    T = Pn[t_idx]                                   # N x 3 truths
    Ys = np.zeros((N, 3, 2), dtype=int)
    for j in range(2):
        p1 = np.where(T == 1, nested['se'][j], 1 - nested['sp'][j])
        Ys[:, :, j] = (nrng.random((N, 3)) < p1).astype(int)
    ff = fit(Ys, ALL, restarts=3, iters=1500, seed=500 + b)
    nn = fit(Ys, nested['patterns'], restarts=3, iters=1500, seed=900 + b)
    if ff and nn: null_stats.append(2 * (ff['ll'] - nn['ll']))
obs = 2 * (full['ll'] - nested['ll'])
out['LRT_full_vs_nested']['p_parametric_bootstrap'] = round((1 + sum(s >= obs for s in null_stats)) / (1 + len(null_stats)), 4) if null_stats else None

# ── 5. distance and relation type under J2 ───────────────────────────────────
dist = np.array([pool[i]['trigger_position'] - pool[i]['cue_position'] for i in items])
near = dist <= 96
for which in ('J1', 'J2', 'both'):
    L = labels(which)
    band = {}
    for nm, mask in (('near_le96', near), ('beyond', ~near)):
        Lm = L[mask]
        acc = {s: wilson(int(Lm[:, k].sum()), int(mask.sum())) for k, s in enumerate(SYS)}
        ps = {f'{a}-{b}': round(mcnemar_exact(Lm[:, SYS.index(a)], Lm[:, SYS.index(b)])[2], 4) for a, b in (('letta', 'mem0'), ('zep', 'mem0'), ('letta', 'zep'))}
        band[nm] = {'n': int(mask.sum()), 'acc': acc, 'p_exact': ps}
    edges = ((1, 96), (97, 184), (185, 273), (274, 393), (394, 690))  # the paper's distance bands
    band['distance_bands'] = {f'{lo}-{hi}': [round(100 * float(L[(dist >= lo) & (dist <= hi), k].mean()), 1) for k in range(3)] + [int(((dist >= lo) & (dist <= hi)).sum())] for lo, hi in edges}
    rel = defaultdict(list)
    for n_, i in enumerate(items):
        rel[pool[i]['relation_type']].append(n_)
    band['relation'] = {r: [round(100 * float(L[idx, k].mean()), 1) for k in range(3)] + [len(idx)] for r, idx in sorted(rel.items())}
    out['by_distance_' + which] = band

cost = [json.loads(l).get('cost_usd_list') or 0 for l in open(os.path.join(os.path.dirname(HERE), 'data', 'second_judge_verdicts.jsonl'))]
out['j2_list_cost_usd'] = round(sum(cost), 2)
json.dump(out, open(os.path.join(os.path.dirname(HERE), 'results', 'second_judge_analysis.json'), 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
