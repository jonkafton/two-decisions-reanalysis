// The texts the registered study fixes, and the two judges that are run through the
// Claude Code CLI: the benchmark's locked cognitive judge prompt (checked against the
// companion repository's file by hash), the answer step's one rule, and the use
// judge's prompt. These are the harness's, not any system's: every system under
// test is answered and judged with the same text.

import { createHash } from 'crypto'
import { mkdtempSync } from 'fs'
import { resolve } from 'path'
import { tmpdir } from 'os'
import { spawn, type ChildProcessWithoutNullStreams } from 'child_process'

// ── the second judge: the benchmark's locked prompt ───────────────────────────

// LoCoMo-Plus's cognitive judge prompt (Appendix B), byte-identical to the copy
// locked in the appliedAI companion repository (judge_prompts.py). Asserted below,
// and again against the fetched file by judges.ts.
export const COGNITIVE_AWARENESS_PROMPT = `You are a Memory Awareness Judge. Determine if the model prediction
demonstrates awareness of the memory cue found in the Evidence.

Scenario: Evidence contains a specific user memory; Question is a trigger interacting
with that memory.

Labels:
- "correct": explicitly acknowledges or adapts to the Memory/Cue (proves recall)
- "wrong": completely ignores the Evidence and gives a generic response.

Evidence (Cue): {cue_dialogue}
Question (Trigger): {trigger_query}
Model Prediction: {model_response}

Return your judgment strictly in JSON format: {{"label": "...", "reason": "..."}}`
export const LOCKED_SHA = '7174feb3462a2636df0c2632da7634f78ea850e9bf5069ec2ceea607d9551ca1'
if (createHash('sha256').update(COGNITIVE_AWARENESS_PROMPT).digest('hex') !== LOCKED_SHA) {
  throw new Error('judge prompt drifted from the locked hash')
}

export function judgePrompt(cue: string, trigger: string, reply: string): string {
  // Python str.format semantics: {{ }} are literal braces in the TEMPLATE, and the
  // substituted text is inserted as-is (function replacers, so a "$&" in a reply
  // is not read as a replacement pattern).
  return COGNITIVE_AWARENESS_PROMPT
    .replace(/\{\{/g, '{').replace(/\}\}/g, '}')
    .replace('{cue_dialogue}', () => cue)
    .replace('{trigger_query}', () => trigger)
    .replace('{model_response}', () => reply)
}

export const SECOND_JUDGE_MODEL = 'claude-opus-5-5'
const NEUTRAL_DIR = mkdtempSync(resolve(tmpdir(), 'judge2-neutral-'))

function parseLabel(raw: string): 'correct' | 'wrong' | null {
  let text = raw.trim()
  if (text.startsWith('```')) text = text.split('\n').filter(l => !l.trim().startsWith('```')).join('\n').trim()
  try {
    const obj = JSON.parse(text) as { label?: string }
    const l = String(obj.label ?? '').trim().toLowerCase()
    if (l === 'correct' || l === 'wrong') return l
  } catch { /* fall through */ }
  const m = text.match(/"label"\s*:\s*"(correct|wrong)"/i)
  return m ? (m[1]!.toLowerCase() as 'correct' | 'wrong') : null
}

/**
 * One verdict from the second judge: no tools, no MCP servers, no user settings, no
 * system prompt, an empty working directory, effort medium. The API keys are removed
 * from its environment so it runs on the subscription, not the API.
 */
export function secondJudge(prompt: string, timeoutMs = 240_000): Promise<{ label: 'correct' | 'wrong' | null; raw: string; error?: string }> {
  return cliJudge(prompt, parseLabel, timeoutMs)
}

// ── the answer step's one rule ────────────────────────────────────────────────
//
// Appended to the answer model's minimal system prompt in every arm of every run.
// Without it the answer model opened 29 to 44 of 100 trigger replies and 12 to 20 of
// 100 near-control replies on the development set by narrating what the memory did
// or did not hold, and that narration, not the cue's use, moved the near false-alarm
// figure by 8 points between two runs of the same system. It bans the shape, not
// the words, because a wording that named "memory, notes, records" was echoed back
// as "the stored background doesn't bear on this".
export const ANSWER_RULE = `Your reply begins with the answer: the first sentence answers the question that was asked. Where something you know about the person bears on the question, let it shape the answer the way your own knowledge would, without saying where it came from. Where it does not bear on the question, it does not appear in the reply at all: no remark on what applies or does not apply, and no reference to anything held about the person, anywhere in the reply.`

// ── the use judge: acted on the fact, or only mentioned it ────────────────────
//
// The locked judge labels a reply "correct" when it "explicitly acknowledges or
// adapts to" the cue, so a reply that names the memory in order to set it aside
// counts the same as one whose advice the cue changed. On the controls that
// conflates two failures of different weight. This judge splits every "correct"
// into acted and mentioned. It is the harness's, not the benchmark's, and runs the
// same way as the second judge. The registered false-alarm figure is reported split.
export type UseLabel = 'acted' | 'mentioned' | 'neither'

export const USE_JUDGE_PROMPT = `You are checking how a reply treats one fact that an assistant remembers about the person it is talking to.

Fact: {cue}
Question: {trigger}
Reply: {reply}

Labels:
- "acted": the fact changes the reply. The advice, answer, plan or recommendation is shaped by it, relies on it, or is adjusted or withheld because of it.
- "mentioned": the reply refers to the fact, or to the assistant's memory or notes about the person, without the fact changing the answer. This includes saying the fact does not apply, saying that nothing stored covers the question, and listing what the memory holds before answering.
- "neither": the reply does not refer to the fact or to any memory about the person.

Return your judgment strictly in JSON format: {"label": "...", "reason": "..."}`

export function useJudgePrompt(cue: string, trigger: string, reply: string): string {
  return USE_JUDGE_PROMPT.replace('{cue}', () => cue).replace('{trigger}', () => trigger).replace('{reply}', () => reply)
}

function parseUse(raw: string): UseLabel | null {
  let text = raw.trim()
  if (text.startsWith('```')) text = text.split('\n').filter(l => !l.trim().startsWith('```')).join('\n').trim()
  try {
    const l = String((JSON.parse(text) as { label?: string }).label ?? '').trim().toLowerCase()
    if (l === 'acted' || l === 'mentioned' || l === 'neither') return l
  } catch { /* fall through */ }
  const m = text.match(/"label"\s*:\s*"(acted|mentioned|neither)"/i)
  return m ? (m[1]!.toLowerCase() as UseLabel) : null
}

export function useJudge(prompt: string, timeoutMs = 240_000): Promise<{ label: UseLabel | null; raw: string; error?: string }> {
  return cliJudge(prompt, parseUse, timeoutMs)
}

function cliJudge<L>(prompt: string, parse: (raw: string) => L | null, timeoutMs: number): Promise<{ label: L | null; raw: string; error?: string }> {
  const env = { ...process.env }
  delete env.ANTHROPIC_API_KEY; delete env.ANTHROPIC_API_KEY_FALLBACK
  const args = ['--print', '--output-format', 'json', '--model', SECOND_JUDGE_MODEL, '--effort', 'medium',
    '--tools', '', '--strict-mcp-config', '--setting-sources', 'project', '--no-session-persistence', '--system-prompt', '']
  return new Promise(res => {
    // CLAUDE_BIN pins the binary: an auto-update that swaps `claude` mid-run can make
    // the spawn fail or throw; either is one judge error, not the end of the run.
    let p: ChildProcessWithoutNullStreams
    try {
      p = spawn(process.env.CLAUDE_BIN || 'claude', args, { cwd: NEUTRAL_DIR, env, stdio: ['pipe', 'pipe', 'pipe'] })
    } catch (e) {
      return res({ label: null, raw: '', error: `spawn: ${(e as Error).message}` })
    }
    let out = '', err = ''
    p.on('error', e => res({ label: null, raw: '', error: `spawn: ${e.message}` }))
    const timer = setTimeout(() => { p.kill('SIGKILL'); res({ label: null, raw: '', error: 'timeout' }) }, timeoutMs)
    p.stdout.on('data', d => { out += d })
    p.stderr.on('data', d => { err += d })
    p.on('close', () => {
      clearTimeout(timer)
      try {
        const d = JSON.parse(out) as { result?: string; is_error?: boolean }
        if (d.is_error) return res({ label: null, raw: '', error: `cli_error: ${out.slice(-300)}` })
        const raw = d.result ?? ''
        res({ label: parse(raw), raw })
      } catch {
        res({ label: null, raw: '', error: `bad_output: ${(out + ' ' + err).slice(-300)}` })
      }
    })
    p.stdin.end(prompt)
  })
}
