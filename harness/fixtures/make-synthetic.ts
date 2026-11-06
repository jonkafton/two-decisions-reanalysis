// Writes the synthetic two-speaker fixture for the H2 driver's smoke test, in the
// exact JSON shapes of the companion repository's pool.json and LoCoMo's
// locomo10.json: one conversation of six short sessions, two speakers with synthetic
// names, three cognitive items (goal, state, value) and three factual questions of
// different categories (single-hop, temporal, adversarial). The pool items'
// full_dialogue_history, cue_position, trigger_position and stitch_meta are produced
// by the adapter's own stitch, so `locomo-adapter.ts --check` on this folder is a
// round trip. No benchmark text anywhere; every person, place and company is made up.
//
//   npx tsx harness/fixtures/make-synthetic.ts   # writes fixtures/synthetic/{pool,locomo10}.json

import { writeFileSync, mkdirSync } from 'fs'
import { resolve } from 'path'
import { loadLocomo, stitch, formatStitchTime } from '../locomo-adapter'

const OUT = resolve(__dirname, 'synthetic')
mkdirSync(OUT, { recursive: true })

const A = 'Priya', B = 'Marco'
type Turn = { speaker: string; text: string }
const t = (speaker: string, text: string): Turn => ({ speaker, text })

const sessions: Array<{ dateTime: string; turns: Turn[] }> = [
  { dateTime: '2:10 pm on 3 March, 2024', turns: [
    t(A, 'Hey Marco! Good to see you. How was the trip to Porto?'),
    t(B, 'Porto was lovely, mostly rain though. I spent two days in bookshops and ate far too many pastries.'),
    t(A, "That sounds like my kind of holiday. I've been stuck at work on the quarterly report."),
    t(B, 'Ouch. At least the weather here has been kind. Any plans for the weekend?'),
    t(A, 'Just the market and maybe a long walk if it stays dry.'),
  ] },
  { dateTime: '6:45 pm on 17 March, 2024', turns: [
    t(B, 'Guess what, I ran the Riverside half marathon on Sunday!'),
    t(A, 'No way, congratulations! How did it go?'),
    t(B, 'Two hours and four minutes. My knees have opinions about it today.'),
    t(A, "Well earned. I've started a pottery class on Tuesday evenings, by the way. My first bowl looks like a hat."),
    t(B, 'A hat is a start. Bring one to the picnic in May.'),
    t(A, 'Deal.'),
  ] },
  { dateTime: '11:20 am on 2 April, 2024', turns: [
    t(A, "I signed up for the pottery class properly now, a whole term. The teacher, Noor, is brilliant."),
    t(B, "That's great. Did you decide about the car in the end?"),
    t(A, "Not yet. The mechanic says the old one has another year in it, so I'm waiting."),
    t(B, "Sensible. I'm thinking of repainting the flat this month, the hallway especially."),
    t(A, 'Pick a colour that hides scuffs. Trust me.'),
  ] },
  { dateTime: '8:05 pm on 19 April, 2024', turns: [
    t(B, 'Hallway is painted. Sage green, no scuffs so far.'),
    t(A, 'Very grown up. We had a team day at work, a cooking class, which was more fun than I expected.'),
    t(B, 'What did you make?'),
    t(A, 'A ridiculous number of dumplings. Everyone took some home.'),
    t(B, "I'd have volunteered for that. Picnic is still on for the 12th of May?"),
    t(A, 'Still on. Bring the frisbee.'),
  ] },
  { dateTime: '4:30 pm on 12 May, 2024', turns: [
    t(A, 'That picnic was perfect. Thanks for organising the food.'),
    t(B, 'Thanks for the bowl! It does look slightly like a hat, but a very nice hat.'),
    t(A, "I'll take it. Noor says my second one is actually round."),
    t(B, 'Progress. I have booked a week in the mountains for June, finally.'),
    t(A, 'Jealous. Send photos.'),
  ] },
  { dateTime: '9:15 am on 30 May, 2024', turns: [
    t(B, "Morning! I've started cycling to work three days a week. My legs are complaining but the mornings are nice."),
    t(A, "Good for you. I'm on holiday from Friday, two weeks of doing very little."),
    t(B, 'Where are you headed?'),
    t(A, "My aunt's place by the lake. Books, swimming, and no quarterly reports."),
    t(B, 'Perfect. Bring a hat.'),
    t(A, 'I have several now, apparently.'),
  ] },
]

const dia = (si: number, ti: number) => `D${si + 1}:${ti + 1}`
const conversation: Record<string, unknown> = { speaker_a: A, speaker_b: B }
sessions.forEach((s, si) => {
  conversation[`session_${si + 1}_date_time`] = s.dateTime
  conversation[`session_${si + 1}`] = s.turns.map((turn, ti) => ({ speaker: turn.speaker, dia_id: dia(si, ti), text: turn.text }))
})

// Factual questions: category codes as LoCoMo uses them (4 single-hop, 2 temporal, 5 adversarial).
const qa = [
  { question: 'Which race did Marco run in March 2024?', answer: 'the Riverside half marathon', evidence: [dia(1, 0)], category: 4 },
  { question: 'When did Priya start her pottery class?', answer: 'March 2024', evidence: [dia(1, 3)], category: 2 },
  { question: 'What did Marco say about his trip to Japan?', evidence: [dia(0, 1)], category: 5, adversarial_answer: 'He loved the temples in Kyoto' },
]

const locomo = [{ sample_id: 'conv-syn-01', conversation, qa, event_summary: {}, observation: {}, session_summary: {} }]
writeFileSync(resolve(OUT, 'locomo10.json'), JSON.stringify(locomo, null, 2) + '\n')

// Cognitive items: cue (A then B) and a trigger (A) that never names the cue; a time
// gap that places each cue at a different insertion point. query_time is the last
// session plus 7 days (6 June 2024 09:15): "one week later" puts the cue level with the
// last session (after it, by the stable sort); "two months later" before session 3
// (6 April); "six months later" before every session.
const plus = [
  { relation_type: 'goal', time_gap: 'two months later',
    cue_dialogue: "A: I'm putting every spare euro aside for a flat deposit, hoping to buy next spring.\nB: That is a serious goal. Good on you for sticking to it.",
    trigger_query: 'A: My laptop is on its last legs. Should I just go for the top-spec model and be done with it?' },
  { relation_type: 'state', time_gap: 'one week later',
    cue_dialogue: 'A: I sprained my wrist on Sunday, so I am typing one-handed for a few weeks.\nB: Ouch, take it easy and let it heal properly.',
    trigger_query: 'A: Any tips for getting through a long report I have to write this week?' },
  { relation_type: 'value', time_gap: 'six months later',
    cue_dialogue: 'A: I only ever buy second-hand clothes now; new fast fashion just does not sit right with me.\nB: That is a lovely principle to live by.',
    trigger_query: 'A: I need a smart outfit for a wedding next month. Where would you start?' },
]

const [base] = loadLocomo(resolve(OUT, 'locomo10.json'))
const poolItems = plus.map((p, i) => {
  const s = stitch(base!, p)
  return {
    sample_id: `cog_syn_${String(i + 1).padStart(4, '0')}`,
    conversation_id: `conv_syn_${String(i + 1).padStart(4, '0')}`,
    relation_type: p.relation_type,
    full_dialogue_history: s.fullDialogue,
    cue_dialogue: p.cue_dialogue,
    cue_position: s.cuePosition,
    trigger_query: p.trigger_query,
    trigger_position: s.triggerPosition,
    time_gap: p.time_gap,
    stitch_meta: { cue_time: formatStitchTime(s.cueTime), query_time: formatStitchTime(s.queryTime) },
  }
})
writeFileSync(resolve(OUT, 'pool.json'), JSON.stringify(poolItems, null, 2) + '\n')
console.log(`wrote ${OUT}/locomo10.json (1 conversation, ${sessions.length} sessions, ${qa.length} questions) and pool.json (${poolItems.length} items; cue after session ${poolItems.map(p => stitch(base!, p).cueAfterSession).join(', ')})`)
