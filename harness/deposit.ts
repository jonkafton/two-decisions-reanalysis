#!/usr/bin/env tsx
// Build the release set of a registered run: every per-item output with the
// benchmark's text replaced by its SHA-256 and length, and everything the
// pipeline wrote (lookup plans, standing blocks, replies, verdicts, counts) in
// full. A reader with the benchmark files restores any hashed text by hashing
// the candidate string (UTF-8, exactly as the run record held it).
//
//   npx tsx harness/deposit.ts --runs <dir>[,<dir>…] [--merged <dir>] [--h1 <h1 result json>] --out <dir>
//
// What is hashed (field → fieldSha256 + fieldChars): the cue dialogue, trigger
// query and trigger text and the cue turns' text of a cognitive item; the
// question, gold, answer, adversarial answer and evidence turns' text of a
// factual question; the probe text of every recall (the near and far control
// texts are the harness's own and are published in harness/h1-near-variants.json
// and harness/h1-far-controls.json); the text of every lookup result the answer
// model saw (`seen`, and each tool_result block in `messages`); the facts the
// capture judge extracted and the text of every memory write in the ingest
// records. What is kept in full: lookup inputs (the planner's queries), the
// standing block and self-view as read at each recall, every reply, every
// verdict with its raw judge output, every usage and count, meta and summary.
// Dropped: tool schemas (they are the product's), snapshot paths, the states/.

import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync, copyFileSync } from 'node:fs'
import { resolve, basename, join } from 'node:path'

type J = Record<string, unknown>
const args = process.argv.slice(2)
const opt = (k: string) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : undefined }
const runs = (opt('--runs') ?? '').split(',').filter(Boolean)
const merged = opt('--merged')
const h1 = opt('--h1')
const out = opt('--out')
if (!runs.length || !out) { console.error('usage: deposit.ts --runs <dir>[,<dir>…] [--merged <dir>] [--h1 <file>] --out <dir>'); process.exit(2) }

const sha = (s: string) => createHash('sha256').update(s, 'utf8').digest('hex')
const written: string[] = []

/** Replace string fields named in `keys` on `o` by their hash and length. */
function hashFields(o: J, keys: string[]): J {
  const r: J = { ...o }
  for (const k of keys) {
    const v = r[k]
    if (typeof v === 'string') { r[`${k}Sha256`] = sha(v); r[`${k}Chars`] = v.length; delete r[k] }
  }
  return r
}
const hashTurns = (turns: unknown) => Array.isArray(turns) ? turns.map(t => (t && typeof t === 'object') ? hashFields(t as J, ['text']) : t) : turns

function depositItems(items: { cognitive: J[]; factual: J[] }): J {
  return {
    cognitive: items.cognitive.map(i => { const r = hashFields(i, ['cueDialogue', 'triggerQuery', 'triggerText']); r.cueTurns = hashTurns(i.cueTurns); return r }),
    // an adversarial question's gold is the adapter's own fixed sentence, not the benchmark's, so it stays
    factual: items.factual.map(q => { const r = hashFields(q, q.category === 5 ? ['question', 'answer', 'adversarialAnswer'] : ['question', 'gold', 'answer', 'adversarialAnswer']); r.evidenceTurns = hashTurns(q.evidenceTurns); return r }),
  }
}

/** The messages the answer model received: keep the shape and the tool_use inputs, hash text content. */
function depositMessages(msgs: unknown): unknown {
  if (!Array.isArray(msgs)) return msgs
  return msgs.map(m => {
    if (!m || typeof m !== 'object') return m
    const mm = m as J
    if (typeof mm.content === 'string') return { ...mm, contentSha256: sha(mm.content), contentChars: mm.content.length, content: undefined }
    if (!Array.isArray(mm.content)) return mm
    return { ...mm, content: mm.content.map(b => {
      if (!b || typeof b !== 'object') return b
      const bb = b as J
      if (bb.type === 'text' && typeof bb.text === 'string') return { type: 'text', textSha256: sha(bb.text), textChars: bb.text.length }
      if (bb.type === 'tool_result') {
        const c = bb.content
        const text = typeof c === 'string' ? c : Array.isArray(c) ? c.map(x => (x && typeof x === 'object' && typeof (x as J).text === 'string') ? (x as J).text : JSON.stringify(x)).join('\n') : JSON.stringify(c ?? '')
        return { type: 'tool_result', tool_use_id: bb.tool_use_id, contentSha256: sha(text), contentChars: text.length }
      }
      return bb // tool_use blocks: the planner's own inputs, kept
    }) }
  })
}

function depositRecall(r: J): J {
  const d = hashFields(r, ['text', 'seen', 'selfSeen'])
  // the standing block is kept in full, as registered; selfSeen is restored below
  d.selfSeen = r.selfSeen; delete d.selfSeenSha256; delete d.selfSeenChars
  d.messages = depositMessages(r.messages)
  d.tools = Array.isArray(r.tools) ? (r.tools as J[]).map(t => t.name) : r.tools
  return d
}

function depositIngestRow(row: J): J {
  const d: J = { ...row }
  const detail = row.detail as J | undefined
  if (detail && typeof detail === 'object') {
    const dd: J = { ...detail }
    if (Array.isArray(detail.capture)) dd.capture = (detail.capture as J[]).map(c => {
      const cc: J = { ...c }
      if (Array.isArray(c.facts)) { cc.factsSha256 = (c.facts as unknown[]).map(f => typeof f === 'string' ? sha(f) : null); cc.factsChars = (c.facts as unknown[]).map(f => typeof f === 'string' ? f.length : null); delete cc.facts }
      return cc
    })
    if (Array.isArray(detail.writes)) dd.writes = (detail.writes as J[]).map(w => hashFields(w, ['text', 'content', 'entry']))
    d.detail = dd
  }
  return d
}

function readJsonl(p: string): J[] { return readFileSync(p, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l) as J) }
function writeJson(p: string, v: unknown) { mkdirSync(resolve(p, '..'), { recursive: true }); writeFileSync(p, JSON.stringify(v, null, 1)); written.push(p) }
function writeJsonl(p: string, rows: unknown[]) { mkdirSync(resolve(p, '..'), { recursive: true }); writeFileSync(p, rows.map(r => JSON.stringify(r)).join('\n') + '\n'); written.push(p) }
function copy(p: string, to: string) { mkdirSync(resolve(to, '..'), { recursive: true }); copyFileSync(p, to); written.push(to) }

function depositRun(dir: string, to: string, withIngest: boolean) {
  const items = JSON.parse(readFileSync(join(dir, 'items.json'), 'utf8')) as { cognitive: J[]; factual: J[] }
  writeJson(join(to, 'items.json'), depositItems(items))
  writeJsonl(join(to, 'recalls.jsonl'), readJsonl(join(dir, 'recalls.jsonl')).map(depositRecall))
  writeJsonl(join(to, 'replies.jsonl'), readJsonl(join(dir, 'replies.jsonl')))
  writeJsonl(join(to, 'verdicts.jsonl'), readJsonl(join(dir, 'verdicts.jsonl')))
  for (const f of ['meta.json', 'summary.json', 'p2_result.json']) if (existsSync(join(dir, f))) copy(join(dir, f), join(to, f))
  if (withIngest) {
    if (existsSync(join(dir, 'base-ingest.json'))) writeJson(join(to, 'base-ingest.json'), (JSON.parse(readFileSync(join(dir, 'base-ingest.json'), 'utf8')) as J[]).map(b => { const r = { ...b }; delete r.snapshots; return r }))
    if (existsSync(join(dir, 'item-prep.json'))) copy(join(dir, 'item-prep.json'), join(to, 'item-prep.json'))
    const ing = join(dir, 'ingest')
    if (existsSync(ing)) for (const f of readdirSync(ing).filter(f => f.endsWith('.jsonl'))) writeJsonl(join(to, 'ingest', f), readJsonl(join(ing, f)).map(depositIngestRow))
  }
}

for (const r of runs) depositRun(r, join(out, basename(r)), true)
if (merged) depositRun(merged, join(out, basename(merged)), false)
if (h1) copy(h1, join(out, 'h1_result.json'))

// A README written beside the set is part of it
if (existsSync(join(out, 'README.md'))) written.push(join(out, 'README.md'))
// SHA256SUMS over everything written, then the set's own digest
const rel = (p: string) => p.slice(resolve(out).length + 1)
const sums = written.map(p => `${sha(readFileSync(p, 'utf8'))}  ${rel(resolve(p))}`).sort((a, b) => a.split('  ')[1]!.localeCompare(b.split('  ')[1]!))
writeFileSync(join(out, 'SHA256SUMS'), sums.join('\n') + '\n')
const setDigest = sha(sums.join('\n') + '\n')
writeFileSync(join(out, 'deposit.sha256'), `${setDigest}  SHA256SUMS\n`)
console.log(`${written.length} files written to ${out}; SHA256SUMS digest ${setDigest}`)
