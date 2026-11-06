// The H1 near-control subset: 25 items per relation type drawn from the 401-item pool
// by a seeded generator, for the near variants Jon hand-checks before registration
// (OSF Study Registration — H1 and H2 Study Designs, "The negative controls").
//
//   npx tsx harness/h1-near-draw.ts            # writes harness/h1-near-subset.json
//
// The generator is mulberry32 seeded with 20261104 (H2's seed), one stream, types in
// the order goal, state, value, causal, items sorted by sample_id within each type, a
// partial Fisher–Yates shuffle taking the first 25. Deterministic; re-running
// reproduces the file byte for byte. Output carries identifiers and the texts the
// variants are written against; the texts are the benchmark's (CC BY-NC 4.0) and the
// file is not redistributed beyond this repository.
import { readFileSync, writeFileSync } from 'fs'
import { resolve } from 'path'
import { createHash } from 'crypto'
import { sourcePaths } from './locomo-adapter'
const REANALYSIS_DIR = resolve(__dirname, '..')

const SEED = 20261104
const PER_TYPE = 25
const TYPES = ['goal', 'state', 'value', 'causal'] as const

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface PoolItem { sample_id: string; relation_type: string; cue_dialogue: string; trigger_query: string; time_gap: string; conversation_id: string }

const pool = JSON.parse(readFileSync(sourcePaths(REANALYSIS_DIR).pool, 'utf8')) as PoolItem[]
const rand = mulberry32(SEED)
const drawn: Array<{ sample_id: string; relation_type: string; conversation_id: string; time_gap: string; cue: string; trigger: string }> = []
for (const t of TYPES) {
  const group = pool.filter(p => p.relation_type === t).sort((a, b) => a.sample_id.localeCompare(b.sample_id))
  for (let i = 0; i < PER_TYPE; i++) {
    const j = i + Math.floor(rand() * (group.length - i))
    ;[group[i], group[j]] = [group[j]!, group[i]!]
  }
  for (const p of group.slice(0, PER_TYPE)) {
    // The cue is speaker A's line of the cue dialogue; the trigger is A's question without its prefix.
    const cue = p.cue_dialogue.split(/\\n|\n/).find(l => l.trim().startsWith('A:'))?.trim().replace(/^A:\s*/, '').replace(/\s*B:.*$/, '') ?? p.cue_dialogue
    drawn.push({ sample_id: p.sample_id, relation_type: p.relation_type, conversation_id: p.conversation_id, time_gap: p.time_gap, cue, trigger: p.trigger_query.replace(/^A:\s*/, '') })
  }
}
const out = resolve(__dirname, 'h1-near-subset.json')
const body = JSON.stringify({ seed: SEED, generator: 'mulberry32, one stream, types goal,state,value,causal, items sorted by sample_id, partial Fisher–Yates, first 25', perType: PER_TYPE, poolSha256: createHash('sha256').update(readFileSync(sourcePaths(REANALYSIS_DIR).pool)).digest('hex'), drawnAt: '2026-11-05', items: drawn }, null, 1)
writeFileSync(out, body)
console.log(`wrote ${out}: ${drawn.length} items; sha256 ${createHash('sha256').update(body).digest('hex').slice(0, 16)}…`)
for (const t of TYPES) console.log(`  ${t}: ${drawn.filter(d => d.relation_type === t).map(d => d.sample_id).join(' ')}`)
