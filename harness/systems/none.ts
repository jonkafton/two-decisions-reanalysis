#!/usr/bin/env tsx
// The reference system: no memory at all. It keeps the sessions it was given, so
// that snapshots are real copies, and answers every recall with nothing. It exists
// so the harness can be run end to end anywhere, and as the smallest example of
// the protocol in system.ts. Its results are the floor the trigger alone sets.
//
//   npx tsx harness/locomo-driver.ts --fixture harness/fixtures/synthetic --system "npx tsx harness/systems/none.ts" --no-judge

import { mkdirSync, cpSync, writeFileSync, readFileSync, existsSync, readdirSync, statSync } from 'fs'
import { resolve } from 'path'
import { serve, type Session } from '../system'

let state: string | null = null
let sessions: Array<{ conversationId: string; session: Session }> = []
const file = () => resolve(state!, 'sessions.json')
const dirSize = (dir: string): number => readdirSync(dir, { withFileTypes: true }).reduce((n, e) => { const p = resolve(dir, e.name); return n + (e.isDirectory() ? dirSize(p) : statSync(p).size) }, 0)

serve({
  hello: () => ({ name: 'none', notes: 'no memory: ingest is kept but never read; recall returns nothing', switches: [] }),
  open(dir, from) {
    if (from) { mkdirSync(dir, { recursive: true }); cpSync(from, dir, { recursive: true }) }
    mkdirSync(dir, { recursive: true })
    state = dir
    sessions = existsSync(file()) ? JSON.parse(readFileSync(file(), 'utf8')) : []
  },
  async ingest(session, conversationId) {
    sessions.push({ conversationId, session })
    writeFileSync(file(), JSON.stringify(sessions))
    return { exchanges: session.exchanges.length, writes: 0, saveRuns: 0, recallCalls: 0, recallFallbacks: 0, usage: {} }
  },
  async recall() {
    return { messages: [], seen: '', standingSeen: null, selfSeen: null, lookups: [], usage: null, faults: [] }
  },
  close() { const bytes = state ? dirSize(state) : 0; state = null; sessions = []; return { bytes } },
  snapshot(from, to) { mkdirSync(to, { recursive: true }); cpSync(from, to, { recursive: true }) },
})
