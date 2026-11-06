// The protocol between the harness and a memory system under test.
//
// The harness owns the design: the item sets, the arms, the controls, the answer
// step, the judges and the scoring. A system owns only its memory: it ingests
// conversations through its own save path and, asked a question, returns what it
// would hand its answer model. A system is any program that speaks this protocol
// on stdin and stdout, one JSON object per line, one response per request, in
// order. Anything it prints on stderr is passed through as its log.
//
//   → {"op":"hello","runDir":"…"}                         ← {"ok":true,"result":SystemInfo}
//   → {"op":"open","state":"<dir>","from":"<dir>"|null}  ← {"ok":true}
//        Open the memory state kept in directory `state`. With `from`, the state is
//        first made a copy of the snapshot in `from`; without it, the directory is
//        opened as it stands (empty when new).
//   → {"op":"ingest","session":Session,"conversationId":"…"}  ← {"ok":true,"result":IngestResult}
//        One session, exchange by exchange, through the system's own save path,
//        with the session's date as "today". Each session is its own conversation.
//   → {"op":"recall","text":"…","switches":Switches}      ← {"ok":true,"result":RecallResult}
//        What the system's answer model would be handed for `text` as the first
//        message of a new conversation, with each manipulated variable at the level
//        given. Nothing is written. A lookup that fails or times out is reported in
//        `faults`; the harness then excludes the item rather than scoring a miss.
//   → {"op":"close"}                                     ← {"ok":true,"result":{"bytes":N}}
//        Close the open state so that its directory is complete and copyable.
//   → {"op":"snapshot","from":"<dir>","to":"<dir>"}       ← {"ok":true}
//        Copy a closed state directory. Copies only; nothing is removed.
//   → {"op":"exit"}                                      ← (the process ends)
//
// A failure is {"ok":false,"error":"…"}; the harness stops on one.
//
// The reference implementation is systems/none.ts, a system with no memory at all.
// The system the registered study runs on is 1Presence's, whose implementation of
// this protocol (presence-system.ts) lives beside its product code and is attached
// to the registration.

import { spawn, type ChildProcess } from 'child_process'
import { createInterface } from 'readline'

export interface Exchange { user: string; assistant: string }
export interface Session { date: string; exchanges: Exchange[] }

export type Level = 'on' | 'off'
/** The manipulated variables. planning: H2's (query formulation before lookup).
 *  standing: H1's (the curated standing block). self: the self-view, off in both
 *  H1 arms and on in production. A system without one of these reports it in
 *  SystemInfo.switches and ignores the level. */
export interface Switches { planning: Level; self: Level; standing: Level }

export interface Tokens { input: number; cacheRead: number; cacheWrite: number; output: number; calls: number }
/** Token usage by stage (a system names its own stages, e.g. "capture-judge", "save-pass"). */
export type Usage = Record<string, Tokens>

export interface SystemInfo {
  name: string
  /** The system's code version, for the run's record. */
  commit?: string
  dirty?: boolean
  /** The models behind each stage, by stage name, with `default` for the rest. */
  models?: Record<string, string>
  /** Which of the three switches the system implements. */
  switches?: Array<keyof Switches>
  notes?: string
}

export interface IngestResult {
  exchanges: number
  /** Memory writes the save path made, however the system counts them. */
  writes: number
  /** Exchanges on which the save path ran (the system's own gate). */
  saveRuns: number
  recallCalls: number
  recallFallbacks: number
  usage: Usage
  /** Anything else the system wants on record for this session (its capture decisions, say). */
  detail?: unknown
}

export interface Fault { at: string; tool?: string; message: string }

export interface RecallResult {
  /** The messages the system appends to the conversation before its answer model
   *  replies: tool_use and tool_result blocks in the Anthropic Messages shape, or
   *  plain text turns. The harness sends them to the answer model as they are. */
  messages: unknown[]
  /** Tool definitions for any tool_use blocks in `messages` (the Anthropic shape). */
  tools?: unknown[]
  /** The text of every lookup result the model sees, joined. */
  seen: string
  standingSeen: string | null
  selfSeen: string | null
  lookups: Array<{ name: string; input: Record<string, unknown>; resultCount: number }>
  metrics?: unknown
  usage: Tokens | null
  faults: Fault[]
}

export type Request =
  | { op: 'hello'; runDir: string }
  | { op: 'open'; state: string; from: string | null }
  | { op: 'ingest'; session: Session; conversationId: string }
  | { op: 'recall'; text: string; switches: Switches }
  | { op: 'close' }
  | { op: 'snapshot'; from: string; to: string }
  | { op: 'exit' }

export type Response = { ok: true; result?: unknown } | { ok: false; error: string }

// ── the client side: what the driver uses ─────────────────────────────────────

export class SystemClient {
  private proc: ChildProcess
  private queue: Array<{ res: (v: unknown) => void; rej: (e: Error) => void }> = []
  private buffer = ''
  private exited: string | null = null

  /** `command` is run through the shell, with the harness's environment. */
  constructor(command: string, opts: { env?: NodeJS.ProcessEnv; cwd?: string } = {}) {
    this.proc = spawn(command, { shell: true, cwd: opts.cwd, env: opts.env ?? process.env, stdio: ['pipe', 'pipe', 'inherit'] })
    this.proc.stdout!.on('data', (chunk: Buffer) => {
      this.buffer += chunk.toString()
      let nl: number
      while ((nl = this.buffer.indexOf('\n')) >= 0) {
        const line = this.buffer.slice(0, nl); this.buffer = this.buffer.slice(nl + 1)
        if (!line.trim()) continue
        const waiting = this.queue.shift()
        if (!waiting) { console.error(`system: unexpected line: ${line.slice(0, 200)}`); continue }
        let r: Response
        try { r = JSON.parse(line) as Response } catch { waiting.rej(new Error(`system: not JSON: ${line.slice(0, 200)}`)); continue }
        if (r.ok) waiting.res(r.result)
        else waiting.rej(new Error(`system: ${r.error}`))
      }
    })
    this.proc.on('exit', (code, signal) => {
      this.exited = `exited (${code ?? signal})`
      for (const w of this.queue.splice(0)) w.rej(new Error(`system ${this.exited} before answering`))
    })
    this.proc.on('error', e => { for (const w of this.queue.splice(0)) w.rej(e) })
  }

  request<T = unknown>(req: Request): Promise<T> {
    if (this.exited) return Promise.reject(new Error(`system ${this.exited}`))
    return new Promise<T>((res, rej) => {
      this.queue.push({ res: v => res(v as T), rej })
      this.proc.stdin!.write(JSON.stringify(req) + '\n')
    })
  }

  hello(runDir: string): Promise<SystemInfo> { return this.request<SystemInfo>({ op: 'hello', runDir }) }
  open(state: string, from: string | null): Promise<void> { return this.request({ op: 'open', state, from }).then(() => undefined) }
  ingest(session: Session, conversationId: string): Promise<IngestResult> { return this.request<IngestResult>({ op: 'ingest', session, conversationId }) }
  recall(text: string, switches: Switches): Promise<RecallResult> { return this.request<RecallResult>({ op: 'recall', text, switches }) }
  close(): Promise<{ bytes: number }> { return this.request<{ bytes: number }>({ op: 'close' }) }
  snapshot(from: string, to: string): Promise<void> { return this.request({ op: 'snapshot', from, to }).then(() => undefined) }
  async exit(): Promise<void> {
    if (this.exited) return
    this.proc.stdin!.write(JSON.stringify({ op: 'exit' }) + '\n')
    await new Promise<void>(res => { const t = setTimeout(() => { this.proc.kill(); res() }, 10_000); this.proc.once('exit', () => { clearTimeout(t); res() }) })
  }
}

// ── the server side: what a system implementation uses ────────────────────────

export interface SystemHandlers {
  hello(runDir: string): Promise<SystemInfo> | SystemInfo
  open(state: string, from: string | null): Promise<void> | void
  ingest(session: Session, conversationId: string): Promise<IngestResult>
  recall(text: string, switches: Switches): Promise<RecallResult>
  close(): Promise<{ bytes: number }> | { bytes: number }
  snapshot(from: string, to: string): Promise<void> | void
  exit?(): Promise<void> | void
}

/** Serve the protocol on stdin/stdout. Requests are handled one at a time, in order. */
export function serve(h: SystemHandlers): void {
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity })
  const reply = (r: Response): void => { process.stdout.write(JSON.stringify(r) + '\n') }
  let chain: Promise<void> = Promise.resolve()
  rl.on('line', line => {
    if (!line.trim()) return
    chain = chain.then(async () => {
      let req: Request
      try { req = JSON.parse(line) as Request } catch { return reply({ ok: false, error: `not JSON: ${line.slice(0, 200)}` }) }
      try {
        switch (req.op) {
          case 'hello': return reply({ ok: true, result: await h.hello(req.runDir) })
          case 'open': await h.open(req.state, req.from); return reply({ ok: true })
          case 'ingest': return reply({ ok: true, result: await h.ingest(req.session, req.conversationId) })
          case 'recall': return reply({ ok: true, result: await h.recall(req.text, req.switches) })
          case 'close': return reply({ ok: true, result: await h.close() })
          case 'snapshot': await h.snapshot(req.from, req.to); return reply({ ok: true })
          case 'exit': await h.exit?.(); rl.close(); process.exit(0)
          default: return reply({ ok: false, error: `unknown op ${String((req as { op: string }).op)}` })
        }
      } catch (e) { reply({ ok: false, error: (e as Error).message }) }
    })
  })
  rl.on('close', async () => { await h.exit?.(); process.exit(0) })
}

// ── token accounting ──────────────────────────────────────────────────────────

export function addUsage(u: Usage, stage: string, t: { inputTokens?: number; cacheReadTokens?: number; cacheCreationTokens?: number; outputTokens?: number }): void {
  const s = (u[stage] ??= { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, calls: 0 })
  s.input += t.inputTokens ?? 0; s.cacheRead += t.cacheReadTokens ?? 0; s.cacheWrite += t.cacheCreationTokens ?? 0; s.output += t.outputTokens ?? 0; s.calls += 1
}
export function mergeUsage(into: Usage, from: Usage): void {
  for (const [stage, t] of Object.entries(from)) {
    const s = (into[stage] ??= { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, calls: 0 })
    s.input += t.input; s.cacheRead += t.cacheRead; s.cacheWrite += t.cacheWrite; s.output += t.output; s.calls += t.calls
  }
}
/** Dollars per million tokens for one model. */
export interface Rate { input: number; output: number; cacheRead: number; cacheWrite: number }
export function costUsd(t: Tokens, r: Rate): number {
  return (t.input * r.input + t.output * r.output + t.cacheRead * r.cacheRead + t.cacheWrite * r.cacheWrite) / 1e6
}

/** Run `tasks` with at most `n` in flight. */
export async function pool<T>(n: number, tasks: Array<() => Promise<T>>): Promise<T[]> {
  const out: T[] = new Array(tasks.length)
  let next = 0
  await Promise.all(Array.from({ length: Math.min(n, tasks.length) }, async () => {
    while (next < tasks.length) { const i = next++; out[i] = await tasks[i]!() }
  }))
  return out
}
