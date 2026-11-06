// LoCoMo / LoCoMo-Plus adapter for the harness. Pure: it reads two JSON files by
// path and produces the harness's Session shape, and it never calls a model.
//
// Inputs (both in this repository's data/source, fetched and hash-checked by
// fetch_data.sh, never redistributed):
//   pool.json      the companion repository's 401-item cognitive pool
//                  (data/cognitive_samples_401_seed42.json)
//   locomo10.json  LoCoMo's ten conversations (snap-research/locomo, CC BY-NC 4.0),
//                  fetched by the same script at a pinned commit
//
// What it reproduces: prepare_data.py's stitch. For every pool item the cue turns
// and the trigger turn are placed among the base conversation's sessions by time
// (cue_time = query_time − time_gap; query_time = last session + 7 days), speakers A
// and B mapped to the base's speaker_a and speaker_b. `checkReproduction` asserts the
// adapter's stitch equals the pool's own full_dialogue_history, cue_position and
// trigger_position, item by item. That is how a pool item is matched to its base.
//
// Speaker mapping is a declared option (SpeakerMapping), not a constant:
//   'speaker'     the trigger's speaker (always A, so speaker_a) is the user; the
//                 other speaker's turns are the assistant's. Turns pair into
//                 exchanges; a session that opens with the other speaker has that
//                 opening folded into the first reply, and a trailing user turn
//                 gets a fixed acknowledgement. Both folds are counted.
//   'transcript'  each session is a transcript the user shares, both speakers named,
//                 in batches of `batchTurns` turns (default 20, the published
//                 runners' INGEST_BATCH_SIZE); the assistant's reply to each batch is a
//                 fixed acknowledgement; the trigger is asked by the user (speaker_a)
//                 in their own voice. The registered mapping.
//
// Nothing here runs an item. The driver (locomo-driver.ts) refuses real data until
// the study is registered.

import { readFileSync } from 'fs'
import { resolve } from 'path'
import { createHash } from 'crypto'
import type { Session, Exchange } from './system'

// ── source shapes ─────────────────────────────────────────────────────────────

export interface LocomoTurn { speaker: string; text: string; diaId?: string }
export interface LocomoSession { index: number; dateTime: Date; date: string; turns: LocomoTurn[] }
export interface LocomoQa {
  index: number
  question: string
  answer: string | null
  adversarialAnswer: string | null
  category: 1 | 2 | 3 | 4 | 5
  evidence: string[]
}
export interface LocomoBase {
  id: string
  speakerA: string
  speakerB: string
  sessions: LocomoSession[]
  qa: LocomoQa[]
  turnCount: number
}

export interface PoolItem {
  sample_id: string
  conversation_id: string
  relation_type: string
  full_dialogue_history: Array<{ speaker: string; text: string }>
  cue_dialogue: string
  cue_position: number
  trigger_query: string
  trigger_position: number
  time_gap: string
  stitch_meta: { cue_time: string; query_time: string }
}

/** LoCoMo's category codes, as its evaluation.py reads them (counts in locomo10:
 *  841 single-hop, 282 multi-hop, 321 temporal, 96 open-domain, 446 adversarial). */
export const CATEGORY_NAME: Record<1 | 2 | 3 | 4 | 5, string> = {
  1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial',
}
/** Which of the companion repository's judge prompts each category takes (judge_prompts.py). */
export type FactualPromptKind = 'fact' | 'temporal' | 'adversarial'
export const PROMPT_FOR_CATEGORY: Record<1 | 2 | 3 | 4 | 5, FactualPromptKind> = {
  1: 'fact', 2: 'temporal', 3: 'fact', 4: 'fact', 5: 'adversarial',
}
/** The reference answer the adversarial judge prompt is given: LoCoMo scores an
 *  adversarial question correct when the reply says the event is not in the
 *  conversation (evaluation.py: "no information available" / "not mentioned"). */
export const ADVERSARIAL_GOLD = 'No information available: the conversation does not mention this; the model should say so rather than answer.'

// ── parsing, mirroring prepare_data.py ───────────────────────────────────────

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

/** `_parse_locomo_session_time`: "%I:%M %p on %d %B, %Y", e.g. "1:56 pm on 8 May, 2023". Read as UTC. */
export function parseLocomoTime(s: string): Date {
  const m = s.trim().match(/^(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})$/i)
  if (!m) throw new Error(`unparseable LoCoMo time: ${JSON.stringify(s)}`)
  let hour = Number(m[1]) % 12
  if (m[3]!.toLowerCase() === 'pm') hour += 12
  const month = MONTHS.findIndex(x => x.toLowerCase() === m[5]!.toLowerCase())
  if (month < 0) throw new Error(`unparseable month in LoCoMo time: ${JSON.stringify(s)}`)
  return new Date(Date.UTC(Number(m[6]), month, Number(m[4]), hour, Number(m[2])))
}

const NUM_WORD: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12 }

/** `_parse_time_gap`: "two months later" → 60 days; weeks ×7, months ×30, years ×365; unparsed → 0. */
export function parseTimeGap(timeGap: string): number {
  const m = timeGap.toLowerCase().trim().match(/\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\b\s*(week|weeks|month|months|year|years)\b/)
  if (!m) return 0
  const num = m[1]!, unit = m[2]!
  const count = /^\d+$/.test(num) ? Number(num) : num === 'a' || num === 'an' ? 1 : NUM_WORD[num] ?? 0
  if (unit.startsWith('week')) return count * 7
  if (unit.startsWith('month')) return count * 30
  if (unit.startsWith('year')) return count * 365
  return 0
}

/** `_parse_ab_dialogue`: lines starting "A:" / "B:"; anything else is dropped. */
export function parseAbDialogue(text: string): Array<{ speaker: 'A' | 'B'; text: string }> {
  const out: Array<{ speaker: 'A' | 'B'; text: string }> = []
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (line.startsWith('A:')) out.push({ speaker: 'A', text: line.slice(2).trim() })
    else if (line.startsWith('B:')) out.push({ speaker: 'B', text: line.slice(2).trim() })
  }
  return out
}

const pad = (n: number) => String(n).padStart(2, '0')
/** Python's strftime('%Y-%m-%d %H:%M') on the UTC-read time. */
export function formatStitchTime(d: Date): string {
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`
}
export function isoDate(d: Date): string {
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`
}
const DAY_MS = 86_400_000

// ── loading ───────────────────────────────────────────────────────────────────

export function loadLocomo(path: string): LocomoBase[] {
  const raw = JSON.parse(readFileSync(path, 'utf8')) as Array<Record<string, unknown>>
  if (!Array.isArray(raw)) throw new Error(`${path}: expected a list of conversations`)
  return raw.map((c, ci) => {
    const conv = c.conversation as Record<string, unknown>
    const sessions: LocomoSession[] = []
    for (let i = 1; conv[`session_${i}`] !== undefined; i++) {
      const dateTime = parseLocomoTime(String(conv[`session_${i}_date_time`]))
      const turns = (conv[`session_${i}`] as Array<Record<string, unknown>>).map(t => ({
        speaker: String(t.speaker), text: String(t.text ?? ''), ...(t.dia_id ? { diaId: String(t.dia_id) } : {}),
      }))
      sessions.push({ index: i, dateTime, date: isoDate(dateTime), turns })
    }
    const qa = ((c.qa as Array<Record<string, unknown>>) ?? []).map((q, index): LocomoQa => ({
      index,
      question: String(q.question ?? ''),
      answer: q.answer === undefined || q.answer === null ? null : String(q.answer),
      adversarialAnswer: q.adversarial_answer === undefined || q.adversarial_answer === null ? null : String(q.adversarial_answer),
      category: Number(q.category) as LocomoQa['category'],
      evidence: Array.isArray(q.evidence) ? (q.evidence as unknown[]).map(String) : [],
    }))
    return {
      id: String(c.sample_id ?? `conv-${ci}`), speakerA: String(conv.speaker_a), speakerB: String(conv.speaker_b),
      sessions, qa, turnCount: sessions.reduce((n, s) => n + s.turns.length, 0),
    }
  })
}

export function loadPool(path: string): PoolItem[] {
  const raw = JSON.parse(readFileSync(path, 'utf8')) as PoolItem[]
  if (!Array.isArray(raw)) throw new Error(`${path}: expected a list of items`)
  return raw
}

/** The two files as the reanalysis repository holds them after ./fetch_data.sh. */
export function sourcePaths(reanalysisDir: string): { pool: string; locomo: string } {
  return { pool: resolve(reanalysisDir, 'data/source/pool.json'), locomo: resolve(reanalysisDir, 'data/source/locomo10.json') }
}

export function sha256File(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

// ── the stitch ────────────────────────────────────────────────────────────────

export interface Stitch {
  fullDialogue: LocomoTurn[]
  cuePosition: number
  triggerPosition: number
  cueTime: Date
  queryTime: Date
  /** The cue goes after this many sessions (0 = before the first). */
  cueAfterSession: number
  cueTurns: LocomoTurn[]
  triggerTurns: LocomoTurn[]
}

/** `_build_context`: events (sessions, cue, trigger) sorted by time, stably, sessions first. */
export function stitch(base: LocomoBase, item: Pick<PoolItem, 'cue_dialogue' | 'trigger_query' | 'time_gap'>): Stitch {
  const last = base.sessions[base.sessions.length - 1]
  if (!last) throw new Error(`base ${base.id} has no sessions`)
  const queryTime = new Date(last.dateTime.getTime() + 7 * DAY_MS)
  const cueTime = new Date(queryTime.getTime() - parseTimeGap(item.time_gap) * DAY_MS)
  const map = (t: { speaker: 'A' | 'B'; text: string }): LocomoTurn => ({ speaker: t.speaker === 'A' ? base.speakerA : base.speakerB, text: t.text })
  const cueTurns = parseAbDialogue(item.cue_dialogue).map(map)
  const triggerTurns = parseAbDialogue(item.trigger_query).map(map)
  type Ev = { t: number; turns: LocomoTurn[]; tag: 'original' | 'cue' | 'query'; order: number }
  const events: Ev[] = base.sessions.map((s, i) => ({ t: s.dateTime.getTime(), turns: s.turns, tag: 'original' as const, order: i }))
  events.push({ t: cueTime.getTime(), turns: cueTurns, tag: 'cue', order: events.length })
  events.push({ t: queryTime.getTime(), turns: triggerTurns, tag: 'query', order: events.length })
  events.sort((a, b) => a.t - b.t || a.order - b.order) // Python's sort is stable; order breaks ties the same way
  const fullDialogue: LocomoTurn[] = []
  let cuePosition = -1, triggerPosition = -1
  for (const ev of events) {
    if (ev.tag === 'cue') cuePosition = fullDialogue.length
    if (ev.tag === 'query') triggerPosition = fullDialogue.length
    for (const t of ev.turns) fullDialogue.push({ speaker: t.speaker, text: t.text })
  }
  // `_compute_insertion`: the last session whose time ≤ cue_time, counted 1-based here.
  let cueAfterSession = 0
  for (const s of base.sessions) { if (s.dateTime.getTime() <= cueTime.getTime()) cueAfterSession = s.index; else break }
  return { fullDialogue, cuePosition, triggerPosition, cueTime, queryTime, cueAfterSession, cueTurns, triggerTurns }
}

function sameDialogue(a: LocomoTurn[], b: Array<{ speaker: string; text: string }>): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i]!.speaker !== b[i]!.speaker || a[i]!.text !== b[i]!.text) return false
  return true
}

/** The base whose stitch reproduces the item exactly, or null. */
export function matchBase(item: PoolItem, bases: LocomoBase[]): { base: LocomoBase; stitch: Stitch } | null {
  for (const base of bases) {
    const s = stitch(base, item)
    if (s.cuePosition === item.cue_position && s.triggerPosition === item.trigger_position
      && formatStitchTime(s.cueTime) === item.stitch_meta.cue_time && formatStitchTime(s.queryTime) === item.stitch_meta.query_time
      && sameDialogue(s.fullDialogue, item.full_dialogue_history)) return { base, stitch: s }
  }
  return null
}

// ── the items the driver runs ─────────────────────────────────────────────────

export interface CognitiveItem {
  sampleId: string
  baseId: string
  relation: string
  timeGap: string
  /** The judges' Evidence and Question, verbatim from the pool ("A: … / B: …"). */
  cueDialogue: string
  triggerQuery: string
  /** The trigger's text with its "A:" prefix removed: what the user says. */
  triggerText: string
  cueTurns: LocomoTurn[]
  cueDate: string
  queryDate: string
  cueAfterSession: number
  cuePosition: number
  triggerPosition: number
  /** Sessions after the cue, up to the trigger (the suffix a faithful replay ingests). */
  suffixSessions: number
  suffixTurns: number
}

export function buildCognitiveItems(pool: PoolItem[], bases: LocomoBase[]): CognitiveItem[] {
  return pool.map(item => {
    const m = matchBase(item, bases)
    if (!m) throw new Error(`${item.sample_id}: no LoCoMo conversation reproduces this item's stitch`)
    const trig = parseAbDialogue(item.trigger_query)
    if (trig.length !== 1 || trig[0]!.speaker !== 'A') throw new Error(`${item.sample_id}: expected one "A:" trigger line`)
    const after = m.base.sessions.filter(s => s.index > m.stitch.cueAfterSession)
    return {
      sampleId: item.sample_id, baseId: m.base.id, relation: item.relation_type, timeGap: item.time_gap,
      cueDialogue: item.cue_dialogue, triggerQuery: item.trigger_query, triggerText: trig[0]!.text,
      cueTurns: m.stitch.cueTurns, cueDate: isoDate(m.stitch.cueTime), queryDate: isoDate(m.stitch.queryTime),
      cueAfterSession: m.stitch.cueAfterSession, cuePosition: m.stitch.cuePosition, triggerPosition: m.stitch.triggerPosition,
      suffixSessions: after.length, suffixTurns: after.reduce((n, s) => n + s.turns.length, 0),
    }
  })
}

export interface FactualQuestion {
  baseId: string
  qaIndex: number
  /** `${baseId}#${qaIndex}`: the identifier the draw file and the run use. */
  id: string
  category: LocomoQa['category']
  categoryName: string
  prompt: FactualPromptKind
  question: string
  /** The reference answer the judge prompt is given (ADVERSARIAL_GOLD for category 5). */
  gold: string
  /** LoCoMo's own fields, kept for the record. */
  answer: string | null
  adversarialAnswer: string | null
  evidence: string[]
  evidenceTurns: Array<{ diaId: string; session: number; speaker: string; text: string }>
  /** Evidence ids that are not "D<session>:<turn>" (8 of 1,986 in locomo10). */
  unparsedEvidence: string[]
  /** The palace holds the base through this session: the last evidence session, or every session when none parses. */
  scopeThroughSession: number
  /** Asked the day after the base's last session, as LoCoMo questions are retrospective. */
  queryDate: string
  questionSha256: string
}

const DIA_RE = /^D(\d+):(\d+)$/

export function buildFactualQuestions(bases: LocomoBase[]): FactualQuestion[] {
  const out: FactualQuestion[] = []
  for (const base of bases) {
    const turnsById = new Map<string, { session: number; speaker: string; text: string }>()
    for (const s of base.sessions) for (const t of s.turns) if (t.diaId) turnsById.set(t.diaId, { session: s.index, speaker: t.speaker, text: t.text })
    const last = base.sessions[base.sessions.length - 1]!
    const queryDate = isoDate(new Date(last.dateTime.getTime() + DAY_MS))
    for (const q of base.qa) {
      const evidenceTurns: FactualQuestion['evidenceTurns'] = []
      const unparsed: string[] = []
      for (const e of q.evidence) {
        const m = e.match(DIA_RE)
        const t = m ? turnsById.get(e) : undefined
        if (m && t) evidenceTurns.push({ diaId: e, ...t })
        else unparsed.push(e)
      }
      const sessions = evidenceTurns.map(t => t.session)
      const gold = q.category === 5 ? ADVERSARIAL_GOLD : (q.answer ?? '')
      out.push({
        baseId: base.id, qaIndex: q.index, id: `${base.id}#${q.index}`,
        category: q.category, categoryName: CATEGORY_NAME[q.category], prompt: PROMPT_FOR_CATEGORY[q.category],
        question: q.question, gold, answer: q.answer, adversarialAnswer: q.adversarialAnswer,
        evidence: q.evidence, evidenceTurns, unparsedEvidence: unparsed,
        scopeThroughSession: sessions.length ? Math.max(...sessions) : base.sessions.length,
        queryDate, questionSha256: createHash('sha256').update(q.question).digest('hex'),
      })
    }
  }
  return out
}

// ── the harness shape ─────────────────────────────────────────────────────────

export type SpeakerMapping = 'speaker' | 'transcript'

/** The scripted reply to a shared transcript batch, and to a trailing user turn under 'speaker'. */
export const TRANSCRIPT_ACK = "Noted, I've read through it."

export interface MappingOptions {
  mapping: SpeakerMapping
  /** 'transcript' only: turns per shared batch (the published runners ingest 20 at a time). */
  batchTurns?: number
}

export interface MappedSession extends Session {
  /** Folds the 'speaker' mapping made: openings by the other speaker folded into the first reply, acknowledgements appended. */
  folds: { leadingFolded: number; ackAppended: number }
}

function longDate(iso: string): string {
  return new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' })
}

/** One LoCoMo session (or the cue block) as a harness Session. `me` is speaker_a, `other` speaker_b. */
export function sessionAs(
  turns: LocomoTurn[], date: string, me: string, other: string, opts: MappingOptions & { introduce?: boolean; label?: string },
): MappedSession {
  const folds = { leadingFolded: 0, ackAppended: 0 }
  if (opts.mapping === 'transcript') {
    const n = Math.max(1, opts.batchTurns ?? 20)
    const exchanges: Exchange[] = []
    for (let i = 0; i < turns.length; i += n) {
      const lines = turns.slice(i, i + n).map(t => `${t.speaker}: ${t.text}`).join('\n')
      const head = i === 0
        ? `${opts.introduce ? `I'm ${me}. ` : ''}${opts.label ?? `My conversation with ${other} on ${longDate(date)}, as a transcript`}:`
        : `(the same conversation, continued):`
      exchanges.push({ user: `${head}\n\n${lines}`, assistant: TRANSCRIPT_ACK })
    }
    return { date, exchanges, folds }
  }
  // 'speaker': group consecutive turns by speaker, then pair user → assistant.
  const groups: Array<{ user: boolean; text: string }> = []
  for (const t of turns) {
    const user = t.speaker === me
    const last = groups[groups.length - 1]
    if (last && last.user === user) last.text += `\n${t.text}`
    else groups.push({ user, text: t.text })
  }
  const exchanges: Exchange[] = []
  let pendingLeading: string | null = null
  for (let i = 0; i < groups.length; i++) {
    const g = groups[i]!
    if (!g.user) {
      // An opening by the other speaker: fold into the first reply.
      pendingLeading = g.text; folds.leadingFolded++
      continue
    }
    const next = groups[i + 1]
    let assistant: string
    if (next && !next.user) { assistant = next.text; i++ } else { assistant = TRANSCRIPT_ACK; folds.ackAppended++ }
    if (pendingLeading !== null) { assistant = `${pendingLeading}\n${assistant}`; pendingLeading = null }
    exchanges.push({ user: g.text, assistant })
  }
  if (pendingLeading !== null && exchanges.length === 0) {
    // A block that is only the other speaker (a one-line cue by B cannot happen: cues open with A).
    exchanges.push({ user: '(continuing our conversation)', assistant: pendingLeading })
  }
  return { date, exchanges, folds }
}

/** The base's sessions as the harness ingests them, one Session per LoCoMo session. */
export function baseSessionsAs(base: LocomoBase, opts: MappingOptions): MappedSession[] {
  return base.sessions.map((s, i) => sessionAs(s.turns, s.date, base.speakerA, base.speakerB, { ...opts, introduce: i === 0 }))
}

/** The cue block as a Session dated cue_time, inserted after `cueAfterSession`. */
export function cueSessionAs(item: CognitiveItem, base: LocomoBase, opts: MappingOptions): MappedSession {
  return sessionAs(item.cueTurns, item.cueDate, base.speakerA, base.speakerB, {
    ...opts, label: `A short exchange with ${base.speakerB} on ${longDate(item.cueDate)}, as a transcript`,
  })
}

/** The trigger as the user's first message of a new conversation, under either mapping. */
export function triggerAs(item: CognitiveItem): { date: string; text: string } {
  return { date: item.queryDate, text: item.triggerText }
}

// ── the check: reproduce every pool item, and the numbers the study design needs ──

export interface AdapterStats {
  pool: { path: string; sha256: string; items: number }
  locomo: { path: string; sha256: string; conversations: number; sessions: number; turns: number }
  items: {
    matched: number
    unmatched: string[]
    byRelation: Record<string, number>
    itemsPerBase: Record<string, number>
    distinctPrefixes: number
    cueBeforeFirstSession: number
    suffixTurns: { total: number; mean: number; min: number; max: number }
    suffixBatchesOf20: number
  }
  factual: {
    questions: number
    byCategory: Record<string, number>
    withUnparsedEvidence: number
    noParsableEvidence: number
  }
  exchanges: { speaker: { base: number; folds: { leadingFolded: number; ackAppended: number } }; transcript20: { base: number } }
}

export function checkReproduction(poolPath: string, locomoPath: string): AdapterStats {
  const pool = loadPool(poolPath)
  const bases = loadLocomo(locomoPath)
  const unmatched: string[] = []
  const prefixes = new Set<string>()
  const byRelation: Record<string, number> = {}
  const itemsPerBase: Record<string, number> = {}
  const suffix: number[] = []
  let cueBeforeFirst = 0, matched = 0
  for (const item of pool) {
    const m = matchBase(item, bases)
    if (!m) { unmatched.push(item.sample_id); continue }
    matched++
    byRelation[item.relation_type] = (byRelation[item.relation_type] ?? 0) + 1
    itemsPerBase[m.base.id] = (itemsPerBase[m.base.id] ?? 0) + 1
    prefixes.add(`${m.base.id}:${m.stitch.cueAfterSession}`)
    if (m.stitch.cueAfterSession === 0) cueBeforeFirst++
    suffix.push(m.base.sessions.filter(s => s.index > m.stitch.cueAfterSession).reduce((n, s) => n + s.turns.length, 0))
  }
  const factual = buildFactualQuestions(bases)
  const byCategory: Record<string, number> = {}
  for (const q of factual) byCategory[`${q.category} ${q.categoryName}`] = (byCategory[`${q.category} ${q.categoryName}`] ?? 0) + 1
  const folds = { leadingFolded: 0, ackAppended: 0 }
  let speakerEx = 0, transcriptEx = 0
  for (const b of bases) {
    for (const s of baseSessionsAs(b, { mapping: 'speaker' })) { speakerEx += s.exchanges.length; folds.leadingFolded += s.folds.leadingFolded; folds.ackAppended += s.folds.ackAppended }
    for (const s of baseSessionsAs(b, { mapping: 'transcript', batchTurns: 20 })) transcriptEx += s.exchanges.length
  }
  const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)
  return {
    pool: { path: poolPath, sha256: sha256File(poolPath), items: pool.length },
    locomo: { path: locomoPath, sha256: sha256File(locomoPath), conversations: bases.length, sessions: sum(bases.map(b => b.sessions.length)), turns: sum(bases.map(b => b.turnCount)) },
    items: {
      matched, unmatched, byRelation, itemsPerBase, distinctPrefixes: prefixes.size, cueBeforeFirstSession: cueBeforeFirst,
      suffixTurns: { total: sum(suffix), mean: suffix.length ? Math.round(sum(suffix) / suffix.length) : 0, min: suffix.length ? Math.min(...suffix) : 0, max: suffix.length ? Math.max(...suffix) : 0 },
      suffixBatchesOf20: sum(suffix.map(n => Math.ceil(n / 20))),
    },
    factual: {
      questions: factual.length, byCategory,
      withUnparsedEvidence: factual.filter(q => q.unparsedEvidence.length > 0).length,
      noParsableEvidence: factual.filter(q => q.evidenceTurns.length === 0).length,
    },
    exchanges: { speaker: { base: speakerEx, folds }, transcript20: { base: transcriptEx } },
  }
}

// ── CLI ───────────────────────────────────────────────────────────────────────
//   npx tsx harness/locomo-adapter.ts --check                      # this repository's data/source
//   npx tsx harness/locomo-adapter.ts --check --pool <pool.json> --locomo <locomo10.json>
// Reads the files, reproduces every item, prints the stats as JSON. No model call.

if (process.argv[1] && /locomo-adapter\.(ts|js)$/.test(process.argv[1]) && process.argv.includes('--check')) {
  const args = process.argv.slice(2)
  const opt = (f: string) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : undefined }
  const dir = opt('--reanalysis-dir') ?? resolve(__dirname, '..')
  const paths = sourcePaths(dir)
  const stats = checkReproduction(opt('--pool') ?? paths.pool, opt('--locomo') ?? paths.locomo)
  console.log(JSON.stringify(stats, null, 2))
  if (stats.items.unmatched.length > 0) { console.error(`${stats.items.unmatched.length} item(s) not reproduced`); process.exit(1) }
}
