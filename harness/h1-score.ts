#!/usr/bin/env tsx
// The registered rule for prediction 1 (OSF registration xufd9, H1), applied to one or
// more run directories of locomo-driver.ts. Study design: the H1 sections of the
// study registration (see harness/README.md).
//
//   npx tsx harness/h1-score.ts --run <dir>[,<dir>…]     # shards are read together
//
// Reads items.json (relation type, time_gap), recalls.jsonl (the exclusion rule: any
// failed lookup in any arm or probe excludes the item from every arm), verdicts.jsonl
// (both judges per reply, and the use judge's row) and meta.json (which judges ran).
//
// The rule, as the design states it:
//   G_t  = A_on,t − A_off,t   both-pass accuracy per relation type, h1-on against h1-off
//   (a)  D1 = (G_goal + G_state)/2 − (G_value + G_causal)/2 ≥ 5 points
//   (b)  gain on the nearest time_gap quintile > gain on the farthest
//   (c)  FA_on − FA_off ≤ 5 points, FA = share of far and paired control replies both
//        judges pass AND the use judge labels acted
//   supported: (a), (b), (c); failed: D1 ≤ 0 or rise > 5; inconclusive otherwise;
//   not testable as registered if any type has fewer than 100 scored items.
// Decisions rest on the point estimates. Wilson 95% intervals sit beside each share and
// a percentile bootstrap (seed 20261104, 10,000 draws, items resampled within type)
// beside D1 and the rise; neither decides.

import { readFileSync, existsSync } from 'fs'
import { resolve } from 'path'
import { parseTimeGap } from './locomo-adapter'

const args = process.argv.slice(2)
const opt = (f: string) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : undefined }
const runs = (opt('--run') ?? '').split(',').filter(Boolean)
if (!runs.length) { console.error('usage: h1-score.ts --run <dir>[,<dir>…]'); process.exit(2) }

type Probe = 'trigger' | 'far' | 'paired' | 'near'
type Arm = 'system-on' | 'system-off' | 'none' | 'h1-on' | 'h1-off'
interface Item { sampleId: string; relation: string; timeGap: string }
interface RecallRow { track: string; id: string; arm: string; probe?: Probe; excluded: boolean; reason: string | null }
interface VerdictRow { track: string; id: string; arm: Arm; probe?: Probe; judge: 'published' | 'second' | 'use'; label: string | null; flag?: string }

const rows = <T>(dir: string, name: string): T[] => existsSync(resolve(dir, name)) ? readFileSync(resolve(dir, name), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l) as T) : []
const items: Item[] = []
const recalls: RecallRow[] = []
const verdicts: VerdictRow[] = []
let judges: Array<'published' | 'second'> = []
for (const dir of runs) {
  const it = JSON.parse(readFileSync(resolve(dir, 'items.json'), 'utf8')) as { cognitive: Item[] }
  items.push(...it.cognitive)
  recalls.push(...rows<RecallRow>(dir, 'recalls.jsonl').filter(r => r.track === 'cognitive'))
  verdicts.push(...rows<VerdictRow>(dir, 'verdicts.jsonl').filter(v => v.track === 'cognitive'))
  const meta = JSON.parse(readFileSync(resolve(dir, 'meta.json'), 'utf8')) as { design: { judges: string[] } }
  judges = meta.design.judges.filter((j): j is 'published' | 'second' => j === 'published' || j === 'second')
}
const seen = new Set<string>()
for (const i of items) { if (seen.has(i.sampleId)) throw new Error(`${i.sampleId} appears in more than one run`); seen.add(i.sampleId) }

const excluded = new Set(recalls.filter(r => r.excluded).map(r => r.id))
const scored = items.filter(i => !excluded.has(i.sampleId))
const byKey = new Map<string, { published?: string | null; second?: string | null; use?: string | null }>()
for (const v of verdicts) {
  const k = `${v.id}:${v.probe ?? 'trigger'}:${v.arm}`
  const e = byKey.get(k) ?? {}
  if (v.judge === 'use') e.use = v.label; else e[v.judge] = v.label
  byKey.set(k, e)
}
/** Both judges "correct" (or the one judge that ran); null when a verdict is missing. */
function passed(id: string, probe: Probe, arm: Arm): boolean | null {
  const e = byKey.get(`${id}:${probe}:${arm}`)
  if (!e || judges.some(j => e[j] === undefined)) return null
  return judges.every(j => e[j] === 'correct')
}
const acted = (id: string, probe: Probe, arm: Arm) => passed(id, probe, arm) === true && byKey.get(`${id}:${probe}:${arm}`)?.use === 'acted'
const mentioned = (id: string, probe: Probe, arm: Arm) => passed(id, probe, arm) === true && byKey.get(`${id}:${probe}:${arm}`)?.use === 'mentioned'

function wilson(k: number, n: number): [number, number] {
  if (!n) return [0, 0]
  const z = 1.96, p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
  return [Math.max(0, (c - h) / d), Math.min(1, (c + h) / d)]
}
const pts = (x: number) => Math.round(1000 * x) / 10
const share = (ids: Item[], f: (i: Item) => boolean | null) => { const ls = ids.map(f).filter((x): x is boolean => x !== null); return { k: ls.filter(Boolean).length, n: ls.length } }
const fmt = (s: { k: number; n: number }) => { const [lo, hi] = wilson(s.k, s.n); return { k: s.k, n: s.n, pct: s.n ? pts(s.k / s.n) : null, wilson95: s.n ? [pts(lo), pts(hi)] : null } }

const TYPES = ['goal', 'state', 'value', 'causal']
function accuracy(ids: Item[]) {
  const on = share(ids, i => passed(i.sampleId, 'trigger', 'h1-on')), off = share(ids, i => passed(i.sampleId, 'trigger', 'h1-off'))
  return { on: fmt(on), off: fmt(off), gainPoints: on.n && off.n ? pts(on.k / on.n - off.k / off.n) : null }
}
const byType = Object.fromEntries(TYPES.map(t => [t, { scored: scored.filter(i => i.relation === t).length, ...accuracy(scored.filter(i => i.relation === t)) }]))
const g = (t: string) => (byType[t] as { gainPoints: number | null }).gainPoints
const d1 = TYPES.every(t => g(t) !== null) ? Math.round(10 * ((g('goal')! + g('state')!) / 2 - (g('value')! + g('causal')!) / 2)) / 10 : null

// (b) time_gap quintiles over the scored items
const ranked = [...scored].sort((a, b) => parseTimeGap(a.timeGap) - parseTimeGap(b.timeGap) || a.sampleId.localeCompare(b.sampleId))
const q = Math.ceil(ranked.length / 5)
const quintiles = Array.from({ length: 5 }, (_, k) => ranked.slice(k * q, (k + 1) * q)).filter(x => x.length)
const quintileGains = quintiles.map((ids, k) => ({ quintile: k + 1, items: ids.length, daysRange: [parseTimeGap(ids[0]!.timeGap), parseTimeGap(ids[ids.length - 1]!.timeGap)], ...accuracy(ids) }))
const nearestGain = quintileGains[0]?.gainPoints ?? null, farthestGain = quintileGains[quintileGains.length - 1]?.gainPoints ?? null

// (c) false alarms on the registered controls (far and paired), acted; mentioned and unsplit beside
function falseAlarms(probes: Probe[], arm: Arm) {
  const pairs = scored.flatMap(i => probes.map(p => [i, p] as const)).filter(([i, p]) => passed(i.sampleId, p, arm) !== null)
  const n = pairs.length
  return { n, acted: fmt({ k: pairs.filter(([i, p]) => acted(i.sampleId, p, arm)).length, n }), mentioned: fmt({ k: pairs.filter(([i, p]) => mentioned(i.sampleId, p, arm)).length, n }), judged: fmt({ k: pairs.filter(([i, p]) => passed(i.sampleId, p, arm) === true).length, n }) }
}
const faOn = falseAlarms(['far', 'paired'], 'h1-on'), faOff = falseAlarms(['far', 'paired'], 'h1-off')
const rise = faOn.n && faOff.n ? Math.round(10 * ((faOn.acted.pct ?? 0) - (faOff.acted.pct ?? 0))) / 10 : null
const perKind = Object.fromEntries((['far', 'paired', 'near'] as Probe[]).map(p => [p, { 'h1-on': falseAlarms([p], 'h1-on'), 'h1-off': falseAlarms([p], 'h1-off') }]))

// decision
const underHundred = TYPES.filter(t => (byType[t] as { scored: number }).scored < 100)
let decision: string
if (underHundred.length) decision = `not testable as registered: fewer than 100 scored items for ${underHundred.join(', ')}`
else if (d1 === null || rise === null || nearestGain === null || farthestGain === null) decision = 'incomplete: verdicts missing'
else if (d1 <= 0 || rise > 5) decision = 'failed'
else if (d1 >= 5 && nearestGain > farthestGain && rise <= 5) decision = 'supported'
else decision = 'inconclusive'

// bootstrap: items resampled within type, seed 20261104, 10,000 draws
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
}
const rand = mulberry32(20261104)
const draws = { d1: [] as number[], rise: [] as number[] }
if (d1 !== null && rise !== null) {
  const byT = Object.fromEntries(TYPES.map(t => [t, scored.filter(i => i.relation === t)]))
  for (let b = 0; b < 10_000; b++) {
    const sample = TYPES.flatMap(t => { const pool = byT[t]!; return Array.from({ length: pool.length }, () => pool[Math.floor(rand() * pool.length)]!) })
    const gains = TYPES.map(t => accuracy(sample.filter(i => i.relation === t)).gainPoints ?? 0)
    draws.d1.push((gains[0]! + gains[1]!) / 2 - (gains[2]! + gains[3]!) / 2)
    const on = falseAlarmsOn(sample, 'h1-on'), off = falseAlarmsOn(sample, 'h1-off')
    draws.rise.push(on - off)
  }
}
function falseAlarmsOn(sample: Item[], arm: Arm): number {
  const pairs = sample.flatMap(i => (['far', 'paired'] as Probe[]).map(p => [i, p] as const)).filter(([i, p]) => passed(i.sampleId, p, arm) !== null)
  return pairs.length ? 100 * pairs.filter(([i, p]) => acted(i.sampleId, p, arm)).length / pairs.length : 0
}
const pctl = (xs: number[], p: number) => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); return Math.round(10 * s[Math.min(s.length - 1, Math.floor(p * s.length))]!) / 10 }

const result = {
  runs, items: items.length, excluded: [...excluded], scored: scored.length, judges,
  byType, d1Points: d1, quintileGains, nearestMinusFarthestPoints: nearestGain !== null && farthestGain !== null ? Math.round(10 * (nearestGain - farthestGain)) / 10 : null,
  falseAlarms: { registered: { controls: 'far and paired, acted', 'h1-on': faOn, 'h1-off': faOff, risePoints: rise }, perKind },
  bootstrap: { seed: 20261104, draws: draws.d1.length, d1: [pctl(draws.d1, 0.025), pctl(draws.d1, 0.975)], rise: [pctl(draws.rise, 0.025), pctl(draws.rise, 0.975)] },
  useJudgeFlags: verdicts.filter(v => v.judge === 'use' && v.flag).length,
  decision,
}
console.log(JSON.stringify(result, null, 2))
console.log(`\nH1: ${decision}${d1 !== null ? ` — D1 ${d1} points; nearest − farthest ${result.nearestMinusFarthestPoints}; false-alarm rise ${rise}` : ''}`)
