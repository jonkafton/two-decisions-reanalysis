#!/usr/bin/env tsx
// The shared-base driver for predictions 1 and 2 (OSF registration xufd9; H2: "Improving
// query planning raises factual accuracy more than cognitive accuracy"; H1: "A bounded,
// curated standing state added to a retrieval pipeline raises goal and state accuracy
// more than causal and value accuracy"). Both studies run on the same memory states:
// the write path is identical in every arm, and the arms differ only in what recall
// reads. The system under test is any program that speaks system.ts's protocol; the
// registered run is on 1Presence's memory pipeline through its implementation of it.
//
// Recall arms (per trigger; the switches are sent with every recall):
//   on      planning on,  self-view on,  standing on   — H2's on arm, production; exploratory for H1
//   off     planning off, self-view on,  standing on   — H2's off arm
//   h1-on   planning on,  self-view off, standing on   — H1's on arm
//   h1-off  planning on,  self-view off, standing off  — H1's off arm
// H1's negative controls run in the h1 arms only: far (a fixed list of general questions,
// assigned by seed), paired (another item of the same base and a different type, by seed),
// near (authored variants for the registered subset, from --near-variants). The use judge
// (acted / mentioned / neither) runs on every reply both judges pass.
//
//   # smoke test on the synthetic fixture (no benchmark data), against the no-memory reference system
//   npx tsx harness/locomo-driver.ts --fixture harness/fixtures/synthetic --system "npx tsx harness/systems/none.ts" --no-judge
//
//   # the registered run (refused until --registration names the OSF registration)
//   npx tsx harness/locomo-driver.ts --registration <osf id> --system "<command>" \
//       --factual-sample preregistration/h2/p2_factual_sample.json --near-variants harness/h1-near-variants.json [--shard 1/4]
//
//   --system "<command>"           the system under test, run through the shell (required)
//   --mapping speaker|transcript   speaker mapping (adapter); default transcript
//   --batch-turns 20               transcript batches
//   --stitch suffix-replay|cue-last  how the cue enters a base (see below); default suffix-replay
//   --answer-model haiku|sonnet|opus|<model id>  the answer step's model; default haiku = claude-haiku-5-5 (the registered level)
//   --items cog_0001,…  --questions conv-26#3,…   subsets
//   --far-controls <file>          the far-control list (default harness/h1-far-controls.json)
//   --near-variants <file>         H1's authored near variants ({items:[{sample_id, near}]}); absent → no near probes
//   --no-h1                        H2 only: the two H2 arms, no controls, no use judge
//   --no-judge | --judges published,second
//   --rates <file>                 dollars per million tokens by model, for the cost figure (default harness/rates.json)
//   --out <dir>                    run directory (default: a new OS temp dir)
//
// What a run does, in order:
//   1. One ingest per base conversation through the system's save path, planning on,
//      session by session, a snapshot after every session (so a prefix through any
//      session can be restored). Snapshots are copies; nothing is deleted.
//   2. Per cognitive item: restore the prefix through the cue's insertion point, ingest
//      the cue (dated cue_time), then (suffix-replay) the remaining sessions in order,
//      or (cue-last) restore the whole base and ingest only the cue. Then recall for
//      the trigger once per arm, each as the first message of a new conversation. The
//      arms differ only at the trigger.
//   3. Per factual question: restore the base through its last evidence session, then
//      the same recalls for the question (H2's two arms).
//   4. Answers: the question plus what recall returned, as the system hands it to its
//      answer model, per arm; plus the question alone (the floor). One answer model
//      and one system prompt for every system.
//   5. Both judges on every reply (judges.ts); verdicts.jsonl; a summary. The
//      registered rules are applied by preregistration/h2/p2_score.py (H2) and
//      harness/h1-score.ts (H1).
//
// Every lookup's outcome is recorded by the system: a question whose lookups timed out
// or failed is marked excluded with the reason and never scored as a miss.

import { resolve } from 'path'
import { mkdirSync, mkdtempSync, writeFileSync, appendFileSync, readFileSync, existsSync } from 'fs'
import { tmpdir } from 'os'
import { execFileSync } from 'child_process'
import Anthropic from '@anthropic-ai/sdk'
import { SystemClient, addUsage, mergeUsage, costUsd, pool, type Usage, type Tokens, type Switches, type Fault, type RecallResult, type Rate, type SystemInfo } from './system'
import { ANSWER_RULE, useJudge, useJudgePrompt, type UseLabel } from './prompts'
import {
  loadLocomo, loadPool, sourcePaths, sha256File, buildCognitiveItems, buildFactualQuestions, baseSessionsAs, cueSessionAs, triggerAs,
  type LocomoBase, type CognitiveItem, type FactualQuestion, type SpeakerMapping,
} from './locomo-adapter'
import {
  PublishedJudge, loadCompanionPrompts, fillPrompt, allowedLabels, secondJudgeVerdict, claudeCliVersion, bothPass,
  SECOND_JUDGE_MODEL, type Verdict, type JudgeInput, type CompanionPrompts,
} from './judges'

// ── arguments ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2)
const flag = (f: string) => args.includes(f)
const opt = (f: string) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : undefined }

const REPO = resolve(__dirname, '..')
const fixture = opt('--fixture')
const registration = opt('--registration')
const systemCommand = opt('--system')
if (!systemCommand) { console.error('--system "<command>" names the system under test (harness/systems/none.ts is the reference).'); process.exit(2) }
const mapping = (opt('--mapping') ?? 'transcript') as SpeakerMapping
if (!['speaker', 'transcript'].includes(mapping)) throw new Error('--mapping must be speaker or transcript')
const batchTurns = Number(opt('--batch-turns') ?? 20)
type Stitch = 'suffix-replay' | 'cue-last'
const stitchMode = (opt('--stitch') ?? 'suffix-replay') as Stitch
if (!['suffix-replay', 'cue-last'].includes(stitchMode)) throw new Error('--stitch must be suffix-replay or cue-last')
/** The registered command names the answer model by class (haiku, the registered level); a full model id is taken as given. */
const MODEL_CLASS: Record<string, string> = { haiku: 'claude-haiku-5-5', sonnet: 'claude-sonnet-5-5', opus: 'claude-opus-5-5' }
const answerArg = opt('--answer-model') ?? 'haiku'
const ANSWER_MODEL = MODEL_CLASS[answerArg] ?? answerArg
const judgesWanted = flag('--no-judge') ? [] : (opt('--judges') ?? 'published,second').split(',').filter(Boolean)
const onlyItems = (opt('--items') ?? '').split(',').filter(Boolean)
const onlyQuestions = (opt('--questions') ?? '').split(',').filter(Boolean)
const shard = opt('--shard')
const h1 = !flag('--no-h1')
const farControlsFile = opt('--far-controls') ?? resolve(__dirname, 'h1-far-controls.json')
const nearVariantsFile = opt('--near-variants')
const ratesFile = opt('--rates') ?? resolve(__dirname, 'rates.json')
const rates: Record<string, Rate> = existsSync(ratesFile) ? JSON.parse(readFileSync(ratesFile, 'utf8')) : {}

type Arm = 'on' | 'off' | 'h1-on' | 'h1-off'
const ARMS: Arm[] = h1 ? ['on', 'off', 'h1-on', 'h1-off'] : ['on', 'off']
const H1_ARMS: Arm[] = ['h1-on', 'h1-off']
/** What each recall arm sets; the system reads them on every recall. */
const ARM_SWITCHES: Record<Arm, Switches> = {
  on: { planning: 'on', self: 'on', standing: 'on' },
  off: { planning: 'off', self: 'on', standing: 'on' },
  'h1-on': { planning: 'on', self: 'off', standing: 'on' },
  'h1-off': { planning: 'on', self: 'off', standing: 'off' },
}
type AnswerArm = 'system-on' | 'system-off' | 'none' | 'h1-on' | 'h1-off'
const answerArmOf = (a: Arm): AnswerArm => a === 'on' ? 'system-on' : a === 'off' ? 'system-off' : a
/** The question a recall and reply answer: the item's trigger, or one of H1's three control kinds. */
type Probe = 'trigger' | 'far' | 'paired' | 'near'

// ── the registration guard ────────────────────────────────────────────────────
// Nothing from LoCoMo or LoCoMo-Plus enters a memory state, an answer call or a judge
// call before the study is registered. The fixture path is the only way round it.

if (!fixture && !registration) {
  console.error('Refusing to run benchmark data: pass --registration <OSF registration id> once the study design is registered, or --fixture <dir> for the synthetic smoke test.')
  process.exit(2)
}

// ── data ──────────────────────────────────────────────────────────────────────

const paths = fixture
  ? { pool: resolve(fixture, 'pool.json'), locomo: resolve(fixture, 'locomo10.json') }
  : sourcePaths(REPO)
const sourceDir = resolve(REPO, 'data/source')
const shaFile = resolve(REPO, 'data/SOURCE_SHA256')

const bases = loadLocomo(paths.locomo)
const poolItems = loadPool(paths.pool)
let items = buildCognitiveItems(poolItems, bases)
let questions = buildFactualQuestions(bases)
if (onlyItems.length) items = items.filter(i => onlyItems.includes(i.sampleId))
if (onlyQuestions.length) questions = questions.filter(q => onlyQuestions.includes(q.id))
const sampleFile = opt('--factual-sample')
if (sampleFile) {
  const drawn = JSON.parse(readFileSync(sampleFile, 'utf8')) as { items: Array<{ id: string; question_sha256: string }> }
  const byId = new Map(drawn.items.map(d => [d.id, d.question_sha256]))
  questions = questions.filter(q => byId.has(q.id))
  for (const q of questions) if (byId.get(q.id) !== q.questionSha256) throw new Error(`${q.id}: question text differs from the drawn sample's hash`)
  if (questions.length !== drawn.items.length) throw new Error(`drawn sample names ${drawn.items.length} questions; ${questions.length} found`)
} else if (!fixture) {
  console.error('A real run needs --factual-sample <p2_factual_sample.json> (the registered draw).')
  process.exit(2)
}
if (shard) {
  const [k, n] = shard.split('/').map(Number)
  if (!k || !n || k > n) throw new Error('--shard must be k/n')
  items = items.filter((_, i) => i % n === k - 1)
  questions = questions.filter((_, i) => i % n === k - 1)
}
const basesNeeded = new Map<string, LocomoBase>()
for (const b of bases) if (items.some(i => i.baseId === b.id) || questions.some(q => q.baseId === b.id)) basesNeeded.set(b.id, b)

// ── H1's controls, assigned over the WHOLE pool so a shard does not change them ──
// Seeded generator: mulberry32, seed 20261104 (the registration's seed), one stream,
// items in sample_id order. Far: the list's questions in a seeded shuffle, cycled.
// Paired: a seeded pick among the same base's other items with a different relation
// type (null when the base has none). Near: the authored variant for the item, if any.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
}
interface Controls { far: string | null; paired: { sampleId: string; text: string } | null; near: string | null }
const controls = new Map<string, Controls>()
const controlFiles: Record<string, unknown> = {}
if (h1) {
  const all = buildCognitiveItems(poolItems, bases).sort((a, b) => a.sampleId.localeCompare(b.sampleId))
  const farList = (JSON.parse(readFileSync(farControlsFile, 'utf8')) as { questions: Array<{ text: string }> }).questions.map(q => q.text)
  const near = nearVariantsFile ? new Map((JSON.parse(readFileSync(nearVariantsFile, 'utf8')) as { items: Array<{ sample_id: string; near: string }> }).items.map(i => [i.sample_id, i.near])) : new Map<string, string>()
  controlFiles.far = { path: farControlsFile, sha256: sha256File(farControlsFile), questions: farList.length }
  controlFiles.near = nearVariantsFile ? { path: nearVariantsFile, sha256: sha256File(nearVariantsFile), variants: near.size } : null
  const rand = mulberry32(20261104)
  const order = [...farList.keys()]
  for (let i = 0; i < order.length - 1; i++) { const j = i + Math.floor(rand() * (order.length - i)); [order[i], order[j]] = [order[j]!, order[i]!] }
  all.forEach((it, k) => {
    const siblings = all.filter(o => o.baseId === it.baseId && o.sampleId !== it.sampleId && o.relation !== it.relation)
    const pick = siblings.length ? siblings[Math.floor(rand() * siblings.length)]! : null
    controls.set(it.sampleId, { far: farList.length ? farList[order[k % order.length]!]! : null, paired: pick ? { sampleId: pick.sampleId, text: pick.triggerText } : null, near: near.get(it.sampleId) ?? null })
  })
}

// ── the run directory and the system ──────────────────────────────────────────

const runDir = opt('--out') ?? mkdtempSync(resolve(tmpdir(), 'locomo-run-'))
mkdirSync(runDir, { recursive: true })
const file = (name: string) => resolve(runDir, name)
const appendRow = (name: string, row: unknown) => appendFileSync(file(name), JSON.stringify(row) + '\n')

function gitState(dir: string): { commit: string; dirty: boolean } {
  try {
    const commit = execFileSync('git', ['-C', dir, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()
    const dirty = execFileSync('git', ['-C', dir, 'status', '--porcelain'], { encoding: 'utf8' }).trim().length > 0
    return { commit, dirty }
  } catch { return { commit: 'unknown', dirty: false } }
}

const sys = new SystemClient(systemCommand)
let systemInfo: SystemInfo = { name: 'unknown' }
const meta: Record<string, unknown> = {}

function writeMeta(): void {
  Object.assign(meta, {
    startedAt: meta.startedAt ?? new Date().toISOString(),
    registration: registration ?? null,
    fixture: fixture ?? null,
    data: { pool: { path: paths.pool, sha256: sha256File(paths.pool) }, locomo: { path: paths.locomo, sha256: sha256File(paths.locomo) } },
    harness: { dir: REPO, ...gitState(REPO) },
    system: { command: systemCommand, ...systemInfo },
    models: { answer: ANSWER_MODEL, secondJudge: SECOND_JUDGE_MODEL, claudeCli: judgesWanted.includes('second') ? claudeCliVersion() : null, publishedJudge: meta.publishedJudge ?? null, system: systemInfo.models ?? null },
    design: { studies: h1 ? ['H1', 'H2'] : ['H2'], mapping, batchTurns: mapping === 'transcript' ? batchTurns : null, stitch: stitchMode, arms: ARMS, armSwitches: Object.fromEntries(ARMS.map(a => [a, ARM_SWITCHES[a]])),
      answerArms: h1 ? ['system-on', 'system-off', 'none', 'h1-on', 'h1-off'] : ['system-on', 'system-off', 'none'], probes: h1 ? ['trigger', 'far', 'paired', 'near'] : ['trigger'], controls: controlFiles,
      controlsAssigned: h1 ? { far: [...controls.values()].filter(c => c.far).length, paired: [...controls.values()].filter(c => c.paired).length, near: [...controls.values()].filter(c => c.near).length } : null,
      useJudge: h1 ? 'prompts.ts USE_JUDGE_PROMPT on every reply both judges pass' : null, judges: judgesWanted },
    counts: { bases: basesNeeded.size, cognitiveItems: items.length, factualQuestions: questions.length, shard: shard ?? null },
    node: process.version,
  })
  writeFileSync(file('meta.json'), JSON.stringify(meta, null, 2))
}

// ── 1. base ingest, snapshotted after every session ───────────────────────────

interface BaseIngest { baseId: string; sessions: number; exchanges: number; usage: Usage; recallCalls: number; recallFallbacks: number; writes: number; saveRuns: number; seconds: number; snapshots: string[] }
const snapOf = (baseId: string, through: number) => resolve(runDir, 'states', baseId, `after-s${through}`)

async function ingestBase(base: LocomoBase): Promise<BaseIngest> {
  const t0 = Date.now()
  const work = resolve(runDir, 'states', base.id, 'work')
  const sessions = baseSessionsAs(base, { mapping, batchTurns })
  const usage: Usage = {}
  let recallCalls = 0, recallFallbacks = 0, writes = 0, saveRuns = 0, exchanges = 0
  const snapshots: string[] = []
  // The empty state is the prefix through zero sessions (a cue before the first session).
  await sys.open(work, null)
  await sys.close()
  await sys.snapshot(work, snapOf(base.id, 0)); snapshots.push(snapOf(base.id, 0))
  for (let i = 0; i < sessions.length; i++) {
    const s = sessions[i]!
    await sys.open(work, null)
    try {
      const r = await sys.ingest({ date: s.date, exchanges: s.exchanges }, `${base.id}-s${i + 1}`)
      mergeUsage(usage, r.usage); recallCalls += r.recallCalls; recallFallbacks += r.recallFallbacks
      writes += r.writes; saveRuns += r.saveRuns; exchanges += r.exchanges
      appendRow(`ingest/${base.id}.jsonl`, { session: i + 1, date: s.date, exchanges: r.exchanges, writes: r.writes, saveRuns: r.saveRuns, recallCalls: r.recallCalls, recallFallbacks: r.recallFallbacks, folds: s.folds, detail: r.detail ?? null })
    } finally { await sys.close() }
    await sys.snapshot(work, snapOf(base.id, i + 1)); snapshots.push(snapOf(base.id, i + 1))
    process.stdout.write(`    ${base.id} session ${i + 1}/${sessions.length} (${exchanges} exchanges so far)\n`)
  }
  return { baseId: base.id, sessions: sessions.length, exchanges, usage, recallCalls, recallFallbacks, writes, saveRuns, seconds: Math.round((Date.now() - t0) / 1000), snapshots }
}

// ── 2/3. recall per arm, with the exclusion rule ──────────────────────────────

interface RecallRow {
  track: 'cognitive' | 'factual'; id: string; arm: Arm
  /** Which question this recall was for, and its text (the trigger, or a control). */
  probe: Probe; text: string
  lookups: RecallResult['lookups']; metrics: unknown; seen: string; standingSeen: string | null; selfSeen: string | null
  messages: unknown[]; tools: unknown[]
  usage: Tokens | null
  faults: Fault[]; excluded: boolean; reason: string | null
}

async function recallArms(track: RecallRow['track'], id: string, text: string, arms: Arm[] = ARMS, probe: Probe = 'trigger'): Promise<RecallRow[]> {
  const rows: RecallRow[] = []
  for (const arm of arms) {
    const r = await sys.recall(text, ARM_SWITCHES[arm])
    const excluded = r.faults.length > 0
    rows.push({
      track, id, arm, probe, text, lookups: r.lookups, metrics: r.metrics ?? null, seen: r.seen, standingSeen: r.standingSeen, selfSeen: r.selfSeen,
      messages: r.messages, tools: r.tools ?? [], usage: r.usage, faults: r.faults, excluded,
      reason: excluded ? `lookup failed: ${r.faults.map(f => `${f.tool ?? f.at}: ${f.message.slice(0, 120)}`).join(' | ')}` : null,
    })
  }
  return rows
}

interface ItemPrep { id: string; cueIngest?: { exchanges: number; writes: number; saveRuns: number }; suffix?: { sessions: number; exchanges: number }; usage: Usage; seconds: number; stateBytes: number }

async function runCognitiveItem(item: CognitiveItem): Promise<{ prep: ItemPrep; rows: RecallRow[] }> {
  const t0 = Date.now()
  const base = basesNeeded.get(item.baseId)!
  const dir = resolve(runDir, 'states', 'items', item.sampleId)
  const through = stitchMode === 'cue-last' ? base.sessions.length : item.cueAfterSession
  const usage: Usage = {}
  const prep: ItemPrep = { id: item.sampleId, usage, seconds: 0, stateBytes: 0 }
  await sys.open(dir, snapOf(base.id, through))
  let rows: RecallRow[] = []
  let bytes = 0
  try {
    const cue = cueSessionAs(item, base, { mapping, batchTurns })
    const c = await sys.ingest({ date: cue.date, exchanges: cue.exchanges }, `${item.sampleId}-cue`)
    mergeUsage(usage, c.usage)
    prep.cueIngest = { exchanges: c.exchanges, writes: c.writes, saveRuns: c.saveRuns }
    appendRow(`ingest/items/${item.sampleId}.jsonl`, { stage: 'cue', date: cue.date, exchanges: c.exchanges, writes: c.writes, saveRuns: c.saveRuns, detail: c.detail ?? null })
    if (stitchMode === 'suffix-replay') {
      const suffix = baseSessionsAs(base, { mapping, batchTurns }).filter((_, i) => i + 1 > item.cueAfterSession)
      let ex = 0
      for (let i = 0; i < suffix.length; i++) {
        const s = suffix[i]!
        const r = await sys.ingest({ date: s.date, exchanges: s.exchanges }, `${item.sampleId}-s${item.cueAfterSession + i + 1}`)
        mergeUsage(usage, r.usage); ex += r.exchanges
        appendRow(`ingest/items/${item.sampleId}.jsonl`, { stage: 'suffix', session: item.cueAfterSession + i + 1, date: s.date, exchanges: r.exchanges, writes: r.writes, saveRuns: r.saveRuns, detail: r.detail ?? null })
      }
      prep.suffix = { sessions: suffix.length, exchanges: ex }
    }
    rows = await recallArms('cognitive', item.sampleId, triggerAs(item).text)
    const ctl = controls.get(item.sampleId)
    if (h1 && ctl) {
      if (ctl.far) rows.push(...await recallArms('cognitive', item.sampleId, ctl.far, H1_ARMS, 'far'))
      if (ctl.paired) rows.push(...await recallArms('cognitive', item.sampleId, ctl.paired.text, H1_ARMS, 'paired'))
      if (ctl.near) rows.push(...await recallArms('cognitive', item.sampleId, ctl.near, H1_ARMS, 'near'))
    }
  } finally { bytes = (await sys.close()).bytes }
  prep.seconds = Math.round((Date.now() - t0) / 1000)
  prep.stateBytes = bytes
  return { prep, rows }
}

async function runFactualQuestion(q: FactualQuestion): Promise<{ prep: ItemPrep; rows: RecallRow[] }> {
  const t0 = Date.now()
  const dir = resolve(runDir, 'states', 'factual', q.id.replace('#', '_'))
  await sys.open(dir, snapOf(q.baseId, q.scopeThroughSession))
  let rows: RecallRow[] = []
  let bytes = 0
  // The factual track is H2's alone: its two arms only (H1's arms would be two recalls per question for nothing).
  try { rows = await recallArms('factual', q.id, q.question, ['on', 'off']) } finally { bytes = (await sys.close()).bytes }
  return { prep: { id: q.id, usage: {}, seconds: Math.round((Date.now() - t0) / 1000), stateBytes: bytes }, rows }
}

// ── 4. answers ────────────────────────────────────────────────────────────────

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY, maxRetries: 4 })

function longDate(iso: string): string {
  return new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' })
}

interface ReplyRow { track: RecallRow['track']; id: string; arm: AnswerArm; probe: Probe; text: string; usage: Tokens }

/** The answer call's parameters: a short reply, no extended thinking. Haiku 5 takes
 *  the explicit thinking switch; the budget is 800 tokens, raised by the newer
 *  generation's tokenizer overhead. */
function answerParams(model: string): { max_tokens: number; thinking?: { type: 'disabled' } } {
  const newer = /^claude-(haiku|sonnet|opus)-5/.test(model)
  const max_tokens = newer ? Math.ceil(800 * 1.3) : 800
  return /^claude-haiku-5/.test(model) ? { max_tokens, thinking: { type: 'disabled' } } : { max_tokens }
}

/** One answer: the minimal system prompt, the question, and (system arms) the recall as the system hands it to its answer model. */
async function answer(track: RecallRow['track'], id: string, arm: AnswerArm, probe: Probe, text: string, date: string, recall: RecallRow | null): Promise<ReplyRow> {
  const messages = [{ role: 'user', content: text }, ...(recall?.messages ?? [])] as Anthropic.MessageParam[]
  const usedTools = new Set((recall?.messages ?? []).flatMap(m => Array.isArray((m as { content: unknown }).content) ? (m as { content: unknown[] }).content : [])
    .filter(b => (b as { type: string }).type === 'tool_use').map(b => (b as { name: string }).name))
  const tools = recall && usedTools.size > 0 ? (recall.tools as Anthropic.Tool[]).filter(t => usedTools.has(t.name)) : undefined
  const r = await client.messages.create({
    model: ANSWER_MODEL,
    ...answerParams(ANSWER_MODEL),
    // The one rule the registration's answer step carries (prompts.ts ANSWER_RULE).
    system: `You are a helpful personal assistant. Today is ${longDate(date)}.\n\n${ANSWER_RULE}`,
    messages,
    ...(tools && tools.length ? { tools, tool_choice: { type: 'none' as const } } : {}),
  })
  const usage: Usage = {}
  addUsage(usage, 'answer', { inputTokens: r.usage.input_tokens, outputTokens: r.usage.output_tokens, cacheReadTokens: r.usage.cache_read_input_tokens ?? 0, cacheCreationTokens: r.usage.cache_creation_input_tokens ?? 0 })
  return { track, id, arm, probe, text: r.content.filter(b => b.type === 'text').map(b => (b as { text: string }).text).join('\n').trim(), usage: usage.answer! }
}

// ── 5. judges ─────────────────────────────────────────────────────────────────

interface VerdictRow { track: RecallRow['track']; id: string; arm: AnswerArm; probe: Probe; judge: 'published' | 'second' | 'use'; label: Verdict['label'] | UseLabel; parse?: string; flag?: string; error?: string; reason?: string; raw: string }

function judgeInput(reply: ReplyRow): JudgeInput {
  if (reply.track === 'cognitive') {
    const item = items.find(i => i.sampleId === reply.id)!
    // A control reply is judged against the item's cue as Evidence and the control question as Question,
    // as the development set judges its controls: a "correct" here is a false alarm.
    const question = reply.probe === 'trigger' ? item.triggerQuery : `A: ${controlText(item.sampleId, reply.probe)}`
    return { kind: 'cognitive', cueDialogue: item.cueDialogue, triggerQuery: question, modelResponse: reply.text }
  }
  const q = questions.find(x => x.id === reply.id)!
  return { kind: q.prompt, question: q.question, gold: q.gold, pred: reply.text, evidence: q.evidenceTurns.map(t => `${t.speaker}: ${t.text}`).join('\n') }
}

function controlText(sampleId: string, probe: Probe): string {
  const c = controls.get(sampleId)
  return probe === 'far' ? c?.far ?? '' : probe === 'paired' ? c?.paired?.text ?? '' : probe === 'near' ? c?.near ?? '' : items.find(i => i.sampleId === sampleId)!.triggerText
}
/** Speaker A's line of the cue dialogue: the fact the use judge is asked about. */
function cueLine(item: CognitiveItem): string {
  // Some pool cues carry a literal backslash-n (not a newline) before "B:"; both split.
  return item.cueDialogue.split(/\\n|\n/).find(l => l.trim().startsWith('A:'))?.trim().replace(/^A:\s*/, '').replace(/\s*B:.*$/, '') ?? item.cueDialogue
}

// ── summary ───────────────────────────────────────────────────────────────────

const pct = (n: number, d: number) => d === 0 ? '—' : `${Math.round((1000 * n) / d) / 10}% (${n}/${d})`

/** The rate for a stage: the answer model's for the answer step, the system's model for its stages. */
function rateFor(stage: string): Rate | null {
  const model = stage === 'answer' ? ANSWER_MODEL : (systemInfo.models?.[stage] ?? systemInfo.models?.default)
  return model && rates[model] ? rates[model]! : null
}
const usdOf = (stage: string, t: Tokens): number | null => { const r = rateFor(stage); return r ? Number(costUsd(t, r).toFixed(4)) : null }

function summarise(recalls: RecallRow[], replies: ReplyRow[], verdicts: VerdictRow[], usage: Usage, recallUsageByArm: Record<Arm, Tokens>, prepUsage: Usage): Record<string, unknown> {
  const excludedIds = new Set(recalls.filter(r => r.excluded).map(r => `${r.track}:${r.id}`))
  const byKey = new Map<string, Partial<Record<'published' | 'second', Verdict>> & { use?: UseLabel | null }>()
  for (const v of verdicts) {
    const k = `${v.track}:${v.id}:${v.probe ?? 'trigger'}:${v.arm}`
    const e = byKey.get(k) ?? {}
    if (v.judge === 'use') e.use = v.label as UseLabel | null
    else e[v.judge] = { label: v.label as Verdict['label'], raw: v.raw, flag: v.flag, error: v.error }
    byKey.set(k, e)
  }
  const judgesRun = judgesWanted as Array<'published' | 'second'>
  const passed = (track: RecallRow['track'], id: string, arm: AnswerArm, probe: Probe = 'trigger'): boolean | null => {
    const e = byKey.get(`${track}:${id}:${probe}:${arm}`)
    if (!e || judgesRun.some(j => !e[j])) return null
    if (judgesRun.length === 2) return bothPass(e.published, e.second)
    return e[judgesRun[0]!]?.label === 'correct'
  }
  const useOf = (track: RecallRow['track'], id: string, arm: AnswerArm, probe: Probe): UseLabel | null | undefined => byKey.get(`${track}:${id}:${probe}:${arm}`)?.use
  const trackSummary = (track: RecallRow['track'], ids: string[]) => {
    const scored = ids.filter(id => !excludedIds.has(`${track}:${id}`))
    const acc = (arm: AnswerArm) => { const ls = scored.map(id => passed(track, id, arm)).filter((x): x is boolean => x !== null); return { n: ls.length, correct: ls.filter(Boolean).length, text: pct(ls.filter(Boolean).length, ls.length) } }
    const perJudge = (judge: 'published' | 'second', arm: AnswerArm) => { const ls = scored.map(id => byKey.get(`${track}:${id}:trigger:${arm}`)?.[judge]?.label).filter(l => l !== undefined); return pct(ls.filter(l => l === 'correct').length, ls.length) }
    const on = acc('system-on'), off = acc('system-off'), none = acc('none')
    return {
      items: ids.length, excluded: ids.length - scored.length,
      bothPass: { 'system-on': on.text, 'system-off': off.text, none: none.text },
      gainOnMinusOffPoints: on.n && off.n ? Math.round(1000 * (on.correct / on.n - off.correct / off.n)) / 10 : null,
      perJudge: Object.fromEntries(judgesRun.map(j => [j, { 'system-on': perJudge(j, 'system-on'), 'system-off': perJudge(j, 'system-off'), none: perJudge(j, 'none') }])),
    }
  }
  const cog = trackSummary('cognitive', items.map(i => i.sampleId))
  const fac = trackSummary('factual', questions.map(q => q.id))
  // H1 preview: both-pass accuracy per relation type in the h1 arms, and the three controls' false-alarm
  // rates, judged (both judges "correct") and acted (the use judge). Point estimates; the registered rule
  // and its intervals are h1-score.ts.
  let h1Preview: Record<string, unknown> | null = null
  if (h1) {
    const scoredIds = items.map(i => i.sampleId).filter(id => !excludedIds.has(`cognitive:${id}`))
    const accBy = (arm: AnswerArm, ids: string[]) => { const ls = ids.map(id => passed('cognitive', id, arm)).filter((x): x is boolean => x !== null); return { n: ls.length, correct: ls.filter(Boolean).length } }
    const byType = Object.fromEntries(['goal', 'state', 'value', 'causal'].map(t => {
      const ids = scoredIds.filter(id => items.find(i => i.sampleId === id)!.relation === t)
      const on = accBy('h1-on', ids), off = accBy('h1-off', ids), prod = accBy('system-on', ids)
      return [t, { scored: ids.length, 'h1-on': pct(on.correct, on.n), 'h1-off': pct(off.correct, off.n), gainPoints: on.n && off.n ? Math.round(1000 * (on.correct / on.n - off.correct / off.n)) / 10 : null, production: pct(prod.correct, prod.n) }]
    }))
    const fa = (probe: Exclude<Probe, 'trigger'>, arm: AnswerArm) => {
      const ids = scoredIds.filter(id => controls.get(id)?.[probe])
      const judged = ids.map(id => passed('cognitive', id, arm, probe)).filter((x): x is boolean => x !== null)
      const acted = ids.filter(id => passed('cognitive', id, arm, probe) && useOf('cognitive', id, arm, probe) === 'acted').length
      const mentioned = ids.filter(id => passed('cognitive', id, arm, probe) && useOf('cognitive', id, arm, probe) === 'mentioned').length
      return { controls: ids.length, judged: pct(judged.filter(Boolean).length, judged.length), acted: pct(acted, judged.length), mentioned: pct(mentioned, judged.length) }
    }
    const acted = (arm: AnswerArm) => { const ids = scoredIds.filter(id => passed('cognitive', id, arm)); return { passed: ids.length, acted: ids.filter(id => useOf('cognitive', id, arm, 'trigger') === 'acted').length, mentionedOnly: ids.filter(id => useOf('cognitive', id, arm, 'trigger') === 'mentioned').length } }
    h1Preview = {
      byType, triggerUse: { 'h1-on': acted('h1-on'), 'h1-off': acted('h1-off'), 'system-on': acted('system-on') },
      falseAlarms: Object.fromEntries((['far', 'paired', 'near'] as const).map(k => [k, { 'h1-on': fa(k, 'h1-on'), 'h1-off': fa(k, 'h1-off') }])),
      note: 'point estimates; the registered rule (a) D1, (b) the time_gap quintiles and (c) the acted false-alarm rise are h1-score.ts',
    }
  }
  const cost = Object.fromEntries(Object.entries(usage).map(([stage, t]) => [stage, { ...t, usd: usdOf(stage, t) }]))
  const usds = Object.values(cost).map(c => (c as { usd: number | null }).usd)
  const totalUsd = usds.every(u => u !== null) ? Number(usds.reduce((s, u) => s + (u as number), 0).toFixed(4)) : null
  let gemini: Record<string, unknown> | null = null
  if (judgesWanted.includes('published')) {
    const n = verdicts.filter(v => v.judge === 'published' && v.label !== null).length
    const promptTokens = Math.round(replies.reduce((s, r) => s + (judgeInputChars(r) / 4), 0))
    const rate = rates['gemini-2.5-flash']
    gemini = { verdicts: n, estimate: 'prompt chars/4 in, 512 out at the gemini-2.5-flash rate (the Judge class returns no usage)', usdUpperBound: rate ? Number(((promptTokens * rate.input + n * 512 * rate.output) / 1e6).toFixed(4)) : null }
  }
  return {
    run: { dir: runDir, fixture: fixture ?? null, registration: registration ?? null, system: systemInfo.name, mapping, stitch: stitchMode, answerModel: ANSWER_MODEL, judges: judgesWanted },
    cognitive: cog, factual: fac, h1Preview,
    h2Preview: cog.gainOnMinusOffPoints !== null && fac.gainOnMinusOffPoints !== null
      ? { factualGainMinusCognitiveGainPoints: Math.round(10 * (fac.gainOnMinusOffPoints - cog.gainOnMinusOffPoints)) / 10, note: 'point estimate only; the registered rule and the bootstrap interval are p2_score.py' }
      : null,
    exclusions: recalls.filter(r => r.excluded).map(r => ({ track: r.track, id: r.id, arm: r.arm, probe: r.probe, reason: r.reason })),
    recallPerArm: Object.fromEntries(ARMS.map(a => { const t = recallUsageByArm[a]; return [a, { ...t, usd: usdOf('trigger-recall', t), inputPerRecall: t.calls ? Math.round((t.input + t.cacheRead + t.cacheWrite) / t.calls) : 0 }] })),
    verdictFlags: verdicts.filter(v => v.flag).length, verdictErrors: verdicts.filter(v => v.error).length,
    cost: { ...cost, totalUsd, geminiJudge: gemini, secondJudge: 'Claude Code CLI on the subscription; no API spend',
      prepUsdIncluded: Object.entries(prepUsage).every(([s]) => rateFor(s)) ? Number(Object.entries(prepUsage).reduce((s, [stage, t]) => s + costUsd(t, rateFor(stage)!), 0).toFixed(4)) : null,
      rates: ratesFile },
  }
}
function judgeInputChars(r: ReplyRow): number { const j = judgeInput(r); return JSON.stringify(j).length }

// ── main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  systemInfo = await sys.hello(runDir)
  writeMeta()
  writeFileSync(file('items.json'), JSON.stringify({ cognitive: items.map(i => ({ ...i })), factual: questions.map(q => ({ ...q })) }, null, 1))
  console.log(`${h1 ? 'H1+H2' : 'H2'} driver — ${fixture ? 'SMOKE on synthetic fixture' : `REGISTERED RUN ${registration}`} — system ${systemInfo.name} — ${basesNeeded.size} base(s), ${items.length} cognitive item(s), ${questions.length} factual question(s) — mapping ${mapping}${mapping === 'transcript' ? ` (batches of ${batchTurns})` : ''}, stitch ${stitchMode}, answers ${ANSWER_MODEL}, judges [${judgesWanted.join(', ')}]\n  run dir ${runDir}`)
  mkdirSync(file('ingest/items'), { recursive: true })
  const usage: Usage = {}
  const prepUsage: Usage = {}

  console.log('  1. base ingest (planning on), snapshot after every session')
  const baseIngests: BaseIngest[] = []
  for (const base of basesNeeded.values()) {
    const b = await ingestBase(base)
    baseIngests.push(b); mergeUsage(usage, b.usage)
    writeFileSync(file('base-ingest.json'), JSON.stringify(baseIngests, null, 1))
  }

  console.log(`  2. ${items.length} cognitive item(s): prefix + cue${stitchMode === 'suffix-replay' ? ' + suffix' : ' (cue last on the whole base)'}, then recall per arm`)
  const recalls: RecallRow[] = []
  const preps: ItemPrep[] = []
  for (const item of items) {
    const { prep, rows } = await runCognitiveItem(item)
    preps.push(prep); mergeUsage(usage, prep.usage); mergeUsage(prepUsage, prep.usage)
    for (const r of rows) { recalls.push(r); appendRow('recalls.jsonl', r) }
    writeFileSync(file('item-prep.json'), JSON.stringify(preps, null, 1))
    process.stdout.write(`    ${item.sampleId}: ${rows.map(r => `${r.arm} ${r.lookups.length} lookup(s)${r.excluded ? ' EXCLUDED' : ''}`).join(', ')} (${prep.seconds}s)\n`)
  }
  console.log(`  3. ${questions.length} factual question(s): base through the evidence session, recall per arm`)
  for (const q of questions) {
    const { prep, rows } = await runFactualQuestion(q)
    preps.push(prep)
    for (const r of rows) { recalls.push(r); appendRow('recalls.jsonl', r) }
    process.stdout.write(`    ${q.id} (${q.categoryName}): ${rows.map(r => `${r.arm} ${r.lookups.length} lookup(s)${r.excluded ? ' EXCLUDED' : ''}`).join(', ')}\n`)
  }
  await sys.exit()
  const recallUsageByArm = Object.fromEntries(ARMS.map(a => [a, { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, calls: 0 }])) as Record<Arm, Tokens>
  for (const r of recalls) {
    const t = recallUsageByArm[r.arm]
    if (r.usage) { t.input += r.usage.input; t.cacheRead += r.usage.cacheRead; t.cacheWrite += r.usage.cacheWrite; t.output += r.usage.output; t.calls += r.usage.calls }
    else t.calls += 1
  }
  mergeUsage(usage, { 'trigger-recall': ARMS.reduce((t, a) => { const u = recallUsageByArm[a]; return { input: t.input + u.input, cacheRead: t.cacheRead + u.cacheRead, cacheWrite: t.cacheWrite + u.cacheWrite, output: t.output + u.output, calls: t.calls + u.calls } }, { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, calls: 0 }) })

  console.log(`  4. answers: ${h1 ? 'system-on, system-off, none; h1-on, h1-off on the trigger and the controls' : 'system-on, system-off, none'}`)
  const jobs: Array<() => Promise<ReplyRow>> = []
  const dateOf = (track: RecallRow['track'], id: string) => track === 'cognitive' ? items.find(i => i.sampleId === id)!.queryDate : questions.find(q => q.id === id)!.queryDate
  // The exclusion rule: an item with a failed lookup in ANY arm or probe is out of every arm.
  const excludedIds = new Set(recalls.filter(r => r.excluded).map(r => `${r.track}:${r.id}`))
  const seen = new Set<string>()
  for (const r of recalls) {
    if (excludedIds.has(`${r.track}:${r.id}`)) continue
    const date = dateOf(r.track, r.id)
    jobs.push(() => answer(r.track, r.id, answerArmOf(r.arm), r.probe, r.text, date, r))
    const k = `${r.track}:${r.id}`
    if (r.probe === 'trigger' && !seen.has(k)) { seen.add(k); jobs.push(() => answer(r.track, r.id, 'none', 'trigger', r.text, date, null)) }
  }
  const replies = await pool(4, jobs)
  for (const r of replies) { appendRow('replies.jsonl', r); addUsage(usage, 'answer', { inputTokens: r.usage.input, outputTokens: r.usage.output, cacheReadTokens: r.usage.cacheRead, cacheCreationTokens: r.usage.cacheWrite }) }

  const verdicts: VerdictRow[] = []
  if (judgesWanted.length) {
    console.log(`  5. judging ${replies.length} replies with [${judgesWanted.join(', ')}]`)
    const prompts: CompanionPrompts = loadCompanionPrompts(sourceDir, shaFile)
    let published: PublishedJudge | null = null
    if (judgesWanted.includes('published')) {
      published = new PublishedJudge({ sourceDir, shaFile, workers: 4 })
      await published.start()
      meta.publishedJudge = published.ready
      writeMeta()
    }
    const record = (r: ReplyRow, judge: VerdictRow['judge'], v: Verdict) => {
      const row: VerdictRow = { track: r.track, id: r.id, arm: r.arm, probe: r.probe, judge, label: v.label, parse: v.parse, flag: v.flag, error: v.error, reason: v.reason, raw: (v.raw ?? '').slice(0, 600) }
      if (v.label === null && !v.error) { row.label = 'wrong'; row.flag = row.flag ?? 'unparseable, recorded wrong' }
      verdicts.push(row); appendRow('verdicts.jsonl', row)
    }
    const tasks: Array<() => Promise<void>> = []
    for (const r of replies) {
      const input = judgeInput(r)
      const labels = allowedLabels(input.kind)
      if (published) tasks.push(async () => {
        let v = await published!.judge(input)
        for (let attempt = 1; v.error && attempt < 3; attempt++) { await new Promise(res => setTimeout(res, 10_000 * attempt)); v = await published!.judge(input) }
        record(r, 'published', v)
      })
      if (judgesWanted.includes('second')) tasks.push(async () => record(r, 'second', await secondJudgeVerdict(fillPrompt(prompts, input), labels)))
    }
    let done = 0
    await pool(6, tasks.map(t => async () => { await t(); if (++done % 10 === 0) console.log(`    ${done}/${tasks.length}`) }))
    if (published) await published.close()

    if (h1) {
      // 5b. The use judge on every cognitive reply both judges passed: acted on the cue, mentioned it, or neither.
      const judgesRun = judgesWanted as Array<'published' | 'second'>
      const labelOf = (r: ReplyRow, judge: 'published' | 'second') => verdicts.find(v => v.track === r.track && v.id === r.id && v.arm === r.arm && v.probe === r.probe && v.judge === judge)
      const passedReplies = replies.filter(r => r.track === 'cognitive' && judgesRun.every(j => labelOf(r, j)?.label === 'correct'))
      console.log(`  5b. use judge on ${passedReplies.length} replies both judges passed`)
      let doneUse = 0
      await pool(6, passedReplies.map(r => async () => {
        const item = items.find(i => i.sampleId === r.id)!
        const prompt = useJudgePrompt(cueLine(item), controlText(r.id, r.probe), r.text)
        let v: Awaited<ReturnType<typeof useJudge>> = { label: null, raw: '' }
        for (let attempt = 0; attempt < 4 && !v.label; attempt++) {
          v = await useJudge(prompt)
          if (!v.label) await new Promise(res => setTimeout(res, (/rate|limit|429/i.test(v.error ?? '') ? 60_000 : 5_000) * (attempt + 1)))
        }
        const row: VerdictRow = { track: r.track, id: r.id, arm: r.arm, probe: r.probe, judge: 'use', label: v.label, error: v.error, raw: v.raw.slice(0, 600) }
        if (v.label === null) { row.label = 'mentioned'; row.flag = 'unparseable, recorded mentioned (the conservative side for rule (c))' }
        verdicts.push(row); appendRow('verdicts.jsonl', row)
        if (++doneUse % 10 === 0) console.log(`    ${doneUse}/${passedReplies.length}`)
      }))
    }
  }

  const summary = summarise(recalls, replies, verdicts, usage, recallUsageByArm, prepUsage)
  meta.finishedAt = new Date().toISOString()
  writeMeta()
  writeFileSync(file('summary.json'), JSON.stringify(summary, null, 2))
  console.log('\n' + JSON.stringify(summary, null, 2))
  console.log(`\nrun directory: ${runDir}\nscore H2 with: python3 preregistration/h2/p2_score.py --run ${runDir}${h1 ? `\nscore H1 with: npx tsx harness/h1-score.ts --run ${runDir}` : ''}`)
}

main().catch(async err => { console.error(err); await sys.exit().catch(() => {}); process.exit(1) })
