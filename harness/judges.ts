// The two judges for the harness (locomo-driver.ts), both reading the companion
// repository's prompts from its own fetched file:
//
//   published  the companion repository's `Judge` class and settings, run from the
//              hash-checked files in the reanalysis repository's data/source as a
//              Python subprocess (published_judge.py): gemini-2.5-flash, temperature
//              0, maxOutputTokens 512, its parser. The Gemini key is read from the
//              environment and never printed.
//   second     claude-opus-5-5 through the Claude Code CLI (prompts.ts secondJudge),
//              CLI version recorded.
//
// Cognitive items take the locked cognitive prompt with the cue as Evidence. Factual
// questions take the companion repository's prompt for the question's category
// (fact or commonsense; temporal; adversarial), as PROMPT_FOR_CATEGORY in the adapter
// maps them. Both judges see the same filled prompt. A verdict passes only when both
// judges label it "correct" (bothPass); "partial" is not a pass.

import { spawn, execFileSync, type ChildProcess } from 'child_process'
import { readFileSync, existsSync } from 'fs'
import { resolve } from 'path'
import { createHash } from 'crypto'
import { COGNITIVE_AWARENESS_PROMPT, SECOND_JUDGE_MODEL, secondJudge } from './prompts'
import type { FactualPromptKind } from './locomo-adapter'

export type Label = 'correct' | 'partial' | 'wrong'
export interface Verdict {
  label: Label | null
  raw: string
  reason?: string
  /** How the label was read: the repository's parser path, or ours for the second judge. */
  parse?: 'json' | 'partial-json' | 'invalid'
  /** Recorded and counted, as registration xufd9's H3 rule does: a label the code rejects, or unparseable output. */
  flag?: string
  /** A transient failure after retries: not a verdict. The driver retries or excludes. */
  error?: string
  attempts?: number
}

export type JudgeInput =
  | { kind: 'cognitive'; cueDialogue: string; triggerQuery: string; modelResponse: string }
  | { kind: FactualPromptKind; question: string; gold: string; pred: string; evidence: string }

// ── the companion repository's prompt texts, from its file ────────────────────

export interface CompanionPrompts { cognitive: string; fact: string; temporal: string; adversarial: string; fileSha256: string }

const LOCKED_SHA = '7174feb3462a2636df0c2632da7634f78ea850e9bf5069ec2ceea607d9551ca1'

/** Read the four templates out of judge_prompts.py (hash-checked against SOURCE_SHA256)
 *  so the second judge is filled from the same text the published judge runs. */
export function loadCompanionPrompts(sourceDir: string, shaFile: string): CompanionPrompts {
  const path = resolve(sourceDir, 'judge_prompts.py')
  const bytes = readFileSync(path)
  const sha = createHash('sha256').update(bytes).digest('hex')
  const want = readFileSync(shaFile, 'utf8').split('\n').map(l => l.trim().split(/\s+/)).find(p => p[1] === 'judge_prompts.py')?.[0]
  if (want !== sha) throw new Error(`judge_prompts.py sha256 ${sha} does not match data/SOURCE_SHA256 (${want})`)
  const text = bytes.toString('utf8')
  const take = (name: string): string => {
    const m = text.match(new RegExp(`${name}\\s*=\\s*"""([\\s\\S]*?)"""`))
    if (!m) throw new Error(`${name} not found in judge_prompts.py`)
    return m[1]!
  }
  const cognitive = take('COGNITIVE_AWARENESS_PROMPT')
  if (createHash('sha256').update(cognitive).digest('hex') !== LOCKED_SHA) throw new Error('cognitive prompt drifted from the locked hash')
  if (cognitive !== COGNITIVE_AWARENESS_PROMPT) throw new Error('cognitive prompt differs from the harness copy')
  return { cognitive, fact: take('FACT_COMMONSENSE_JUDGE_PROMPT'), temporal: take('TEMPORAL_JUDGE_PROMPT'), adversarial: take('ADVERSARIAL_JUDGE_PROMPT'), fileSha256: sha }
}

/** Python str.format: `{{`/`}}` are literal braces; `{name}` takes the field as-is. */
export function pyFormat(template: string, fields: Record<string, string>): string {
  return template.replace(/\{\{|\}\}|\{(\w+)\}/g, (m, name: string | undefined) => {
    if (m === '{{') return '{'
    if (m === '}}') return '}'
    if (name !== undefined && name in fields) return fields[name]!
    throw new Error(`template field ${m} has no value`)
  })
}

export function fillPrompt(prompts: CompanionPrompts, input: JudgeInput): string {
  if (input.kind === 'cognitive') {
    return pyFormat(prompts.cognitive, { cue_dialogue: input.cueDialogue, trigger_query: input.triggerQuery, model_response: input.modelResponse })
  }
  const fields: Record<string, string> = { question: input.question, gold: input.gold, pred: input.pred, evidence: input.evidence }
  return pyFormat(prompts[input.kind], fields)
}

export function allowedLabels(kind: JudgeInput['kind']): Label[] {
  return kind === 'fact' ? ['correct', 'partial', 'wrong'] : ['correct', 'wrong']
}

/** Our reading of a judge reply for the second judge: the repository's parser shape,
 *  with "partial" allowed where the prompt offers it, and no default-to-wrong. */
export function parseLabel(raw: string, labels: Label[]): { label: Label | null; reason: string; parse: Verdict['parse'] } {
  let text = raw.trim()
  if (text.startsWith('```')) text = text.split('\n').filter(l => !l.trim().startsWith('```')).join('\n').trim()
  try {
    const obj = JSON.parse(text) as { label?: unknown; reason?: unknown }
    const l = String(obj.label ?? '').trim().toLowerCase() as Label
    if (labels.includes(l)) return { label: l, reason: String(obj.reason ?? ''), parse: 'json' }
  } catch { /* fall through */ }
  const m = text.match(/"label"\s*:\s*"(correct|partial|wrong)"/i)
  if (m && labels.includes(m[1]!.toLowerCase() as Label)) return { label: m[1]!.toLowerCase() as Label, reason: 'partial-json', parse: 'partial-json' }
  return { label: null, reason: '', parse: 'invalid' }
}

// ── the published judge: a pool of Python workers over the companion's own code ──

export interface PublishedJudgeOptions {
  /** The reanalysis repository's data/source and data/SOURCE_SHA256. */
  sourceDir: string
  shaFile: string
  /** Workers (one request in flight each). */
  workers?: number
  /** The interpreter with the companion's two packages (openai, python-dotenv). */
  python?: string
}

export const DEFAULT_JUDGE_PYTHON = resolve(process.env.HOME ?? '', '.cache/two-decisions/h2-judge-venv/bin/python')

/** certifi's CA bundle in the judge's interpreter, or null when it is not installed. */
function certifiBundle(python: string): string | null {
  try { return execFileSync(python, ['-c', 'import certifi; print(certifi.where())'], { encoding: 'utf8' }).trim() || null } catch { return null }
}

interface Worker { proc: ChildProcess; busy: boolean; buffer: string; pending: Map<string, (v: Verdict) => void> }

export class PublishedJudge {
  private workers: Worker[] = []
  private queue: Array<() => void> = []
  private seq = 0
  /** The ready line from the first worker: protocol lock, file hashes, prompt hashes, python version. */
  ready: Record<string, unknown> | null = null
  readonly python: string
  private readonly secret: string

  constructor(private readonly opts: PublishedJudgeOptions) {
    this.python = opts.python ?? process.env.H2_JUDGE_PYTHON ?? DEFAULT_JUDGE_PYTHON
    this.secret = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY ?? ''
  }

  async start(): Promise<void> {
    if (!existsSync(this.python)) {
      throw new Error(`published judge: no Python at ${this.python}. Create it outside the repository:\n`
        + `  python3 -m venv ~/.cache/two-decisions/h2-judge-venv && ~/.cache/two-decisions/h2-judge-venv/bin/pip install "openai>=1.52.0" "python-dotenv>=1.0.1" certifi\n`
        + `or point H2_JUDGE_PYTHON at an interpreter that has them.`)
    }
    if (!this.secret) throw new Error('published judge: GEMINI_API_KEY (or GOOGLE_API_KEY) is not in the environment')
    const n = Math.max(1, this.opts.workers ?? 4)
    const script = resolve(__dirname, 'published_judge.py')
    const certBundle = process.env.SSL_CERT_FILE ?? certifiBundle(this.python)
    const starts: Array<Promise<void>> = []
    for (let i = 0; i < n; i++) {
      const env: NodeJS.ProcessEnv = {
        PATH: process.env.PATH, HOME: process.env.HOME, LANG: process.env.LANG,
        SOURCE_DIR: this.opts.sourceDir, SOURCE_SHA256: this.opts.shaFile,
        GOOGLE_API_KEY: this.secret, OPENAI_API_KEY: 'unused-by-the-published-judge',
        // A python.org build on macOS ships no root certificates, so urllib's TLS
        // verification of the Gemini endpoint fails without a bundle (seen 2026-11-04).
        ...(certBundle ? { SSL_CERT_FILE: certBundle } : {}),
      }
      const proc = spawn(this.python, ['-I', script], { env, stdio: ['pipe', 'pipe', 'pipe'] })
      const w: Worker = { proc, busy: false, buffer: '', pending: new Map() }
      let stderr = ''
      proc.stderr!.on('data', d => { stderr += d })
      starts.push(new Promise<void>((res, rej) => {
        const onFirst = (chunk: Buffer) => {
          w.buffer += chunk.toString()
          const nl = w.buffer.indexOf('\n')
          if (nl < 0) return
          const line = w.buffer.slice(0, nl); w.buffer = w.buffer.slice(nl + 1)
          proc.stdout!.off('data', onFirst)
          try {
            const ready = JSON.parse(line) as Record<string, unknown>
            if (!ready.ready) throw new Error(line)
            this.ready ??= ready
            proc.stdout!.on('data', c => this.onData(w, c))
            res()
          } catch (e) { rej(new Error(`published judge worker did not start: ${this.redact(String(e))}`)) }
        }
        proc.stdout!.on('data', onFirst)
        proc.on('exit', code => {
          if (!this.ready) rej(new Error(`published judge worker exited (${code}): ${this.redact(stderr.slice(-800))}`))
          for (const [, cb] of w.pending) cb({ label: null, raw: '', error: `worker exited (${code}): ${this.redact(stderr.slice(-300))}` })
          w.pending.clear()
        })
        proc.on('error', e => rej(e))
      }))
      this.workers.push(w)
    }
    await Promise.all(starts)
  }

  private redact(s: string): string { return this.secret ? s.split(this.secret).join('[redacted]') : s }

  private onData(w: Worker, chunk: Buffer): void {
    w.buffer += chunk.toString()
    let nl: number
    while ((nl = w.buffer.indexOf('\n')) >= 0) {
      const line = w.buffer.slice(0, nl); w.buffer = w.buffer.slice(nl + 1)
      if (!line.trim()) continue
      let out: Verdict & { id?: string }
      try { out = JSON.parse(line) } catch { continue }
      const cb = out.id ? w.pending.get(out.id) : undefined
      if (!cb) continue
      w.pending.delete(out.id!)
      w.busy = false
      if (out.error) out.error = this.redact(out.error)
      cb(out)
      this.queue.shift()?.()
    }
  }

  private async acquire(): Promise<Worker> {
    for (;;) {
      const free = this.workers.find(w => !w.busy && !w.proc.killed && w.proc.exitCode === null)
      if (free) { free.busy = true; return free }
      await new Promise<void>(res => this.queue.push(res))
    }
  }

  /** One verdict. Transient failures come back as `error`; the caller decides. */
  async judge(input: JudgeInput): Promise<Verdict> {
    const w = await this.acquire()
    const id = `j${++this.seq}`
    const req = input.kind === 'cognitive'
      ? { id, kind: 'cognitive', cue_dialogue: input.cueDialogue, trigger_query: input.triggerQuery, model_response: input.modelResponse }
      : { id, kind: input.kind, question: input.question, gold: input.gold, pred: input.pred, evidence: input.evidence }
    return new Promise<Verdict>(res => {
      w.pending.set(id, v => res({ ...v, raw: v.raw ?? '' }))
      w.proc.stdin!.write(JSON.stringify(req) + '\n')
    })
  }

  async close(): Promise<void> {
    for (const w of this.workers) { try { w.proc.stdin!.end() } catch { /* closed */ } }
    await Promise.all(this.workers.map(w => new Promise<void>(res => {
      if (w.proc.exitCode !== null) return res()
      const t = setTimeout(() => { w.proc.kill(); res() }, 5_000)
      w.proc.on('exit', () => { clearTimeout(t); res() })
    })))
  }
}

// ── the second judge ──────────────────────────────────────────────────────────

let cliVersion: string | null = null
/** `claude --version`, once, for the run's record. */
export function claudeCliVersion(): string {
  if (cliVersion === null) {
    try { cliVersion = execFileSync(process.env.CLAUDE_BIN || 'claude', ['--version'], { encoding: 'utf8' }).trim() }
    catch (e) { cliVersion = `unknown (${(e as Error).message.slice(0, 80)})` }
  }
  return cliVersion
}

export { SECOND_JUDGE_MODEL }

/**
 * One second-judge verdict on a filled prompt, with the harness's call (no tools, no
 * system prompt, an empty directory, effort medium) and retries on rate limits and
 * transient errors. Unparseable output after the attempts is returned with label null
 * and a flag; the driver records it as wrong and flagged, as registration xufd9's H3
 * rule does.
 */
export async function secondJudgeVerdict(prompt: string, labels: Label[], attempts = 4): Promise<Verdict> {
  let last: { raw: string; error?: string } = { raw: '' }
  for (let attempt = 0; attempt < attempts; attempt++) {
    const v = await secondJudge(prompt)
    last = v
    if (v.error) {
      await new Promise(res => setTimeout(res, (/rate|limit|429|overloaded/i.test(v.error ?? '') ? 60_000 : 5_000) * (attempt + 1)))
      continue
    }
    const p = parseLabel(v.raw, labels)
    if (p.label) return { label: p.label, raw: v.raw, reason: p.reason, parse: p.parse, attempts: attempt + 1 }
    await new Promise(res => setTimeout(res, 3_000))
  }
  if (last.error) return { label: null, raw: '', error: last.error, attempts }
  return { label: null, raw: last.raw, parse: 'invalid', flag: `unparsed after ${attempts} attempts`, attempts }
}

/** The registered scoring: a verdict counts only when both judges pass it. */
export function bothPass(published: Verdict | undefined, second: Verdict | undefined): boolean {
  return published?.label === 'correct' && second?.label === 'correct'
}
