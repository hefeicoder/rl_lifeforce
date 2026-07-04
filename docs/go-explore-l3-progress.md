# Go-Explore @ the L3 cave/flame wall — progress & plan

_Last updated: 2026-06-27. Branch: `go-explore`. Captures the session where we
resumed frontier search on the level-3 wall and diagnosed it visually._

## Goal

Get the policy past the **level-3 cave / flame-prominence passage**, where it
reliably dies. Success bar (clarified this session): **"obstacle pass"** — a clean
handoff into survivable territory beyond the passage — **not** a full stage clear.
Stage clear is hundreds of steps further downstream; `cleared=True` in the tools
means *stage* clear and is the wrong target for a single-wall search.

## The Go-Explore loop (how the tooling is meant to chain)

1. `tools/explore_frontier.py` — from a frontier save-state, advance the greedy
   policy to its near-failure point, then random-search a short **bridge** (macro
   holds, ε random vs. stochastic-policy actions) that survives + hands off cleanly
   back to the greedy policy. Scored by **total survival steps** (stage-clear is the
   primary sort key). Outputs a BC demo (`demos/<name>.npz`) + a new frontier
   state (`states/<name>.state`) **only if the best rollout survived the bridge**
   (clean handoff).
2. `tools/self_imitation.py` — behaviour-clone the demo into the policy (small LR,
   few epochs) to seed the maneuver without wrecking the rest of the policy.
3. `python -m src.train --resume <bc-ckpt>` — resume PPO so survival reward
   robustifies it. (self-imitation loop)

Key inputs used this session:
- Policy: `checkpoints/lv1-front-speed/lifeforce_ppo_2000000_steps.zip` (2M steps,
  finished 11:40, 2026-06-27).
- Frontier: `states/l3_wall.state`.

## What we ran (this session) and the results

All from `states/l3_wall.state` with the 2M policy. Greedy baseline from the loaded
frontier = **112 steps** (dies at the wall). Search advances to the near-failure
point (margin = steps-before-death to start from); "local" = greedy baseline from
the effective frontier (~104).

| Run | bridge-steps | candidates | ε / macro / margin | Best (from eff. frontier) | Clean handoff? | Greedy continued after handoff |
|-----|-----|-----|-----|-----|-----|-----|
| greedy | 0 | — | — | 104 | — | — |
| `l3_bridge`        | 60  | 1000 | 0.5 / 12 / 20  | 108 (+4)  | yes | ~48 steps, then dies at same wall |
| `l3_bridge_m40/60/80` | 60 | 3000 ea | 0.7 / 18 / 40,60,80 | **108 (+4) — identical for all 3 margins** | yes | ~48 steps, then dies |
| `l3_bridge_long`   | 150 | 3000 | 0.7 / 18 / 60  | 144 (+40) | **no** — died at step 144 *within* bridge | 0 |
| `l3_pass`          | 120 | 6000 | 0.7 / 18 / 60  | 120 (+16) | **no** — died on the last bridge step | 0 |

**Pattern:** the ship dies at a *fixed position* ~104–144 steps from the frontier.
Nothing passes it — not greedy, not ~13,000 random bridges across margins/lengths.
Longer bridges just let random actions flail deeper before dying; they never hand
off to a survivable state. The 60-step handoffs that *did* save a state put greedy
right back at the same wall (108 ≈ the 104 it dies at alone). **Pure random
frontier search has plateaued on this obstacle** — more candidates / margins /
macro will not help.

## The decisive finding (visual)

Recorded greedy from `states/l3_wall.state`:
`videos/l3_wall_death.mp4` (`ep 0: score=223 steps=112 max_x=16 did not clear`).
Frame montages: `images/l3_wall/l3_wall_trajectory.png` (every-8th frame across the
run) and `images/l3_wall/l3_wall_death.png` (last 14 frames, the death).

**The "wall" is not a hard obstacle or a scripted death — it's the policy refusing
to advance.** The ship stays **pinned against the far-left edge the whole time**
(hence `max_x=16`), dodging the rising blue/orange flame columns *in place* instead
of threading rightward through the narrowing cave passage. It gets cornered between
the scrolling terrain and the bottom flames and dies (explosion is bottom-left in
the final frames).

This explains the search results:
- **Why random bridges plateau:** the required maneuver is a long *sustained
  forward thread* through the whole passage; a 60–150 step random burst can't
  reconstruct it, and the bridge starts with the ship already pinned left.
- **Why handoffs derail (+4):** the greedy policy immediately retreats left again —
  it actively *wants* to camp, undoing any forward progress.

Ties directly to the recent reward work on `go-explore` (front-rush reward,
`MAX_SPEED 2→4`, front-quarter survival bonus, de-baiting the fork): all of it is
the fight to make the ship *advance* instead of camp. `max_x=16` says it still
isn't taking hold at this section.

## Conclusion

**Frontier search is the wrong tool for this wall** — it's fighting the policy's own
retreat-and-camp behavior. Demos it can find are all "die slightly later while
camping," not "advance through." The root cause is the policy not advancing.

## Open decision (next step — pick one)

1. **Train at the wall (PPO).** Resume PPO from `states/l3_wall.state` in the
   curriculum with the front-rush/forward reward, so RL learns to advance through
   the passage. Addresses the root cause (camping) rather than patching one
   maneuver. _(Leading candidate.)_
2. **Hand-author a forward demo.** `tools/capture_state.py` + manual play to record
   a human "thread-the-passage" demo, `self_imitation` it, then resume PPO. Gives
   the policy a forward-motion example random search can't find.
3. **Add CEM to the search.** Implement cross-entropy-method search in
   `explore_frontier` (smarter than random) to reconstruct the long forward
   maneuver. More code; still fights greedy derailment on handoff.
4. **Reconsider the reward.** Step back and inspect why front-rush isn't taking hold
   — review reward terms in `src/config.py` before more training/search.

---

## Session 2 (2026-06-27 cont.) — pivot to reward shaping at the wall

Decided **frontier search is the wrong tool** (it fights the policy's camp-left
prior). Pivoted to: fix instrumentation/curriculum, then RL at the wall. Plan order:
1) isolate curriculum to `l3_wall`, 2) true x diagnostics, 3) honest baseline,
4) furthest-progress ratchet reward + curriculum flags, 5) focused PPO.

### Steps 1–3 done

- **Isolated curriculum:** moved `l2_wall` + all `l3_bridge*` states to
  `states/_archive/` (excluded by the non-recursive `states/*.state` glob). Only
  `states/l3_wall.state` is active. **Do not self-imitate the `l3_bridge*.npz`
  demos — they are anti-demos** (every saved handoff resets at x=14, *left* of the
  x=22 start; the policy moves backward).
- **True x diagnostics** added to `LifeForceWrapper` (so training logs carry them,
  not just play.py): per-episode `max_x` (TRUE furthest, not terminal), `terminal_x`,
  `mean_x`. Fixed `play.py`'s mislabel (it printed terminal `x_pos` as "max_x").
- **Honest baseline from `l3_wall.state`** (2M policy):

  | | max_x (true) | terminal_x | mean_x | steps |
  |---|---|---|---|---|
  | greedy (det) | **29** | 16 | 17 | 112 |
  | stochastic (n=8) | 39–63 (~49) | always 14–18 | 22–28 | 97–112 |

  The ship pokes right then **retreats to the wall (x≈16) every time** and dies.
  The runs that venture furthest die *soonest* (max_x 61/63 → 97 steps vs 112 for
  camping): **advancing currently costs ~15 steps of survival.** That valley is the
  whole problem. (`x_pos` is SCREEN x 15..232, a positioning proxy, not world
  progress — true progress = survival time / `0x2F` scroll clock.)

### Step 4 — reward design (the key decision)

Rejected plain **x-delta** shaping: it telescopes to `(x_final − x_start)`, and since
the ship always ends at x≈16 ≈ start, venturing and camping net the *same* (~0)
x-reward → won't break the basin. Chose an **episode-local furthest-progress
ratchet** instead:

`r_xmax = REWARD_XMAX · max(0, x_frac − best_x_frac_this_episode)`, update best.

- Telescopes to `(max_reached − start)`, not `(final − start)` → pays for the
  EXCURSION, never claws back on retreat, no front-camp bait (holding the front pays
  nothing; only beating your own best does).
- **Episode-local** (best resets each episode) so every worker re-earns the forward
  path from the start state — a global "ever reached" signal would vanish after one
  worker found it.
- Implausible positive jumps (`dx_frac > X_RATCHET_JUMP_CAP=0.25`, e.g. death/screen
  snap) advance best but pay 0.
- **`REWARD_XMAX = 30.0`** (config.py): valley ≈15 steps ≈3.0 alive; greedy→stochastic
  max is ~0.16 normalized, so coeff 30 → ~4.8 for the excursion (> 3.0, learnable)
  without making instant front-suicide dominant. Logged separately as `reward/xmax`.
- `REWARD_XPOS` unchanged (0.05, front-quarter) as a small "hold position" term.
- Smoke check (hold RIGHT from l3_wall): reaches **max_x=107** (ship CAN advance),
  `reward/xmax`=11.75 for the excursion, dies at step 23 → net +6.3 vs camping ~23,
  so suicide-rush is not dominant. Balance confirmed.

**New curriculum flags** (`src/train.py`): `--curriculum-glob` (default all
`states/*.state`) and `--curriculum-mix` — drill one section without shuffling files
(reproducible experiments). Threaded through `make_thunk`/`make_env`/`CurriculumStart`
(now takes a glob pattern, not a dir). New TB metrics: `lifeforce/best_max_x`,
`lifeforce/recent_max_x`.

### Step 5 — focused PPO run (RUNNING)

```
python -m src.train \
  --resume checkpoints/lv1-front-speed/lifeforce_ppo_2000000_steps.zip \
  --run-name l3-wall-xmax --timesteps 500000 --ent-coef 0.03 \
  --curriculum-glob states/l3_wall.state --curriculum-mix 0.9
```
TB run `l3-wall-xmax`. **Judge by true `max_x` AND `steps`:** max_x climbing +
steps recovering = basin escape (success); max_x up but steps collapsing = coeff
30 too high (lower it / add a cap). Baseline to beat: greedy max_x 29, steps 112.

### Code touched this session

`src/config.py` (REWARD_XMAX, X_RATCHET_JUMP_CAP), `src/env.py` (x diagnostics +
ratchet in LifeForceWrapper; CurriculumStart glob; make_env/make_thunk params),
`src/train.py` (curriculum flags, max_x metrics), `src/play.py` (true x labels).
**Uncommitted** as of this writing.

### Step 5 results — ratchet works, then hold-forward gate (deterministic from l3_wall)

| stage | run | max_x | mean_x | steps | read |
|---|---|---|---|---|---|
| baseline | (2M) | 29 | 17 | 112 | camp left |
| 100k | l3-wall-xmax | 71 | 30 | 116 | poke forward |
| 200k | l3-wall-xmax | 82 | 30 | 115 | poke further, still camp |
| +100k | l3-wall-hold | 148 | 88 | 48 | over-shoot: rush+die (value-fn lag) |
| **+200k** | **l3-wall-hold** | **141** | **45** | **121** | **settled: hold forward + survive longer** |

- **Ratchet (REWARD_XMAX=30)** pulled the ship out of the camp-left basin: max_x 29→82,
  steps stable. But mean_x stayed ~30 / terminal_x ~16 = "dart to new max, RETREAT,
  camp" — ratchet rewards reaching new ground, not HOLDING it.
- **Lowered hold-gate (X_FRONT_FRAC 0.75→0.25, x≥69)** resuming from the 200k ckpt
  (`--curriculum-mix 1.0`, isolated l3_wall): mean_x 30→45, max_x →141 (deep into the
  passage; the frontier search's deepest survivor was 144), steps 112→**121** (now
  ABOVE baseline). The +100k over-shoot (rush to 148, die at 48) self-corrected —
  reward math: rush-die (~19) < hold-survive (~23), and the value fn was stale right
  after the mid-training gate change.
- `terminal_x` ~16–17 is a **death-snap artifact** (x resets on the death frame),
  NOT a real retreat — it's constant across very different mean_x. Trust max_x /
  mean_x / steps instead.
- **Status:** forward capability solved (plays deep in the passage + survives longer),
  but still dies without a clean pass/handoff beyond the wall. Threading *through* the
  gap is the remaining gap. NOT yet regression-tested on the full level (gauntlet #1).

Checkpoints: `checkpoints/l3-wall-xmax/` (ratchet, to 200k), `checkpoints/l3-wall-hold/`
(hold-gate, +200k → `lifeforce_ppo_final.zip`), `checkpoints/l3-wall-hold2/`
(+300k continuation → 700k total; max_x/steps plateaued ~155/~117, score 223→323).
Config now: REWARD_XMAX=30, X_FRONT_FRAC=0.25, REWARD_XPOS=0.05.

### Continued training (l3-wall-hold2, 700k total) — plateau at a terrain pinch

Deterministic from l3_wall: max_x 141→**155**, mean_x 45→60, steps ~117 (flat),
**score 223→323** (now fighting/collecting deeper, not progressing further — steps =
distance is flat). Video `videos/l3_hold2_death.mp4` (+ montages `images/l3_wall/
hold2_traj.png`, `hold2_death.png`): the ship now **advances through the cave**,
shoots the flame columns, and dies at a **terrain pinch (~x155)** — threading the
top edge past a rising red stalagmite + a dotted serpent, gets cornered, dies. A
precise **Y-navigation death at a specific pinch**, not a camping death.

### Go-Explore re-attempt with the hold2 policy — DID NOT pass (non-reproducible)

Ran `explore_frontier` with the hold2 policy (now that it advances, not retreats).
It reported a clean handoff at **x=202** (`states/l3_pinch.state`), total survival
117→225, looking like a pinch pass. **It is NOT a real pass** — caught by trying to
reproduce/visualize it:
- A clean replay (advance 97 greedy → replay the demo bridge → greedy) reached only
  **max_x=155 and died** — it did NOT reach x=202.
- Determinism probe (each run 2/2 reproducible, so the env IS deterministic fresh):
  greedy from l3_wall = 117 steps / max_x 155 / retreats to x=16 and dies; advance 97
  → x=15 → +20 = 117 (consistent). Greedy from `l3_pinch.state` (x=202) = 37 steps.
- BUT the search reported "local baseline from the effective frontier = **93**" where
  a clean run gives **~20**. That 93-vs-20 gap means **the search's internal rollouts
  ran under accumulated emulator RNG/state that does NOT reproduce from a fresh
  start.** So its "+35 to x=202" exploited non-reproducible internal state.
- **Conclusion: Go-Explore is unreliable on this env** (its determinism assumption
  breaks across save/reload + RNG). This re-validates the earlier pivot away from
  search toward RL. `l3_pinch.state` is a real x=202 snapshot but the policy can NOT
  reproducibly reach it.

### Corrected status (honest)

**The obstacle is NOT passed.** What the session actually achieved: RL moved the
failure point from **x=29 (camping)** to **x=155 (the real pinch)** and the ship now
drives deep + survives as long — but the **same retreat-when-forward-is-dangerous
behavior recurs AT the pinch** (greedy reaches x=155, then retreats and dies at 117
rather than committing through the threading gap). We *relocated* the retreat, we did
not eliminate it. (An earlier note in this session over-claimed a "verified pass" —
that was wrong; the video reconstruction contradicted it and the determinism probe
confirmed no clean pass.)

### Open next step (decide)

Make the pinch itself the training target so the policy must learn to THREAD, not
approach-and-retreat:
1. Capture the state at ~x150 (the moment before retreat) as a new curriculum start
   (`tools/capture_state.py` or programmatic advance), train from there.
2. And/or add **Y-threading shaping** (the death is a vertical-navigation failure at
   the gap).
3. Full-level regression check (gate 0.25 vs gauntlet #1) still pending.
Do NOT rely on Go-Explore here (non-reproducible). Do NOT BC `l3_pinch.npz` as a
"pass" demo — it doesn't reproduce.

## Session 3 — reproducible pinch curriculum drill

Implemented `tools/capture_frontier_states.py`, a deterministic capture helper for
this phase. It does not search or claim a pass; it simply runs a checkpoint from a
known state, records the x/y trace, and saves emulator states at selected lead-ins
or first x-threshold crossings.

Captured from:

```
python -m tools.capture_frontier_states \
  --model checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip \
  --from-state states/l3_wall.state --prefix l3_pinch \
  --before-death 60 45 30 20 10 --x-targets 100 120 140 150
```

Ground truth from the hold2 policy:

| state | step | x | y | keep? |
|---|---:|---:|---:|---|
| `l3_pinch_x100.state` | 12 | 106 | 115 | yes |
| `l3_pinch_x120.state` | 14 | 120 | 115 | yes |
| `l3_pinch_x140.state` | 32 | 141 | 150 | yes |
| `l3_pinch_x150.state` / `maxx` | 35 | 155 | 143 | maybe late, but kept |
| `l3_pinch_bd60.state` | 57 | 57 | 122 | archive; already retreating |
| `l3_pinch_bd45/bd30/bd20/bd10` | 72+ | <=29 | various | archive; back at left wall |

The usable x-threshold states were moved under `states/l3_pinch_curriculum/` and
the bad/non-reproducible artifacts were archived:

- `states/_archive/pinch_capture/l3_pinch.state` (the invalid Go-Explore x=202
  snapshot)
- `states/_archive/pinch_capture/l3_pinch_bd*.state` (left-wall retreat states)
- `demos/_archive/l3_pinch.npz` (invalid non-reproducible "pass" demo)

Then trained a focused PPO drill:

```
python -m src.train \
  --resume checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip \
  --run-name l3-pinch-drill --timesteps 200000 --ent-coef 0.03 \
  --curriculum-glob 'states/l3_pinch_curriculum/*.state' \
  --curriculum-mix 1.0 --save-freq 100000
```

Training result:

- `curr/best_max_x`: 169 -> **183**
- `curr/best_steps`: 107 -> **110**
- no clear/pass; stable PPO metrics, no collapse
- final checkpoint: `checkpoints/l3-pinch-drill/lifeforce_ppo_final.zip`

Deterministic eval before/after:

| start state | hold2 steps / max_x | pinch-drill steps / max_x | read |
|---|---:|---:|---|
| `states/l3_wall.state` | 117 / 155 | **118 / 155** | no transfer; still dies at pinch |
| `x100` | 34 / 169 | **109 / 155** | big local survival fix, but less forward |
| `x120` | 103 / 162 | **107 / 148** | small local survival fix |
| `x140` | 85 / 155 | 84 / 155 | unchanged |
| `x150/maxx` | 82 / 155 | **86 / 155** | tiny local survival fix |

Video: `videos/l3_pinch_drill_death.mp4`
(`steps=118 max_x=155 terminal_x=17 mean_x=59.3` from `l3_wall.state`).

**Status after Session 3:** the pinch drill improved local survival from some
approach states but did **not** teach a reproducible pass through the pinch and did
not move the full local wall-start max_x past 155. The remaining blocker is not
"approach the pinch" anymore; it is choosing the correct vertical threading action
at/near x140-155. More generic forward x shaping is likely spent.

**Next likely lever:** add a general Y/terrain-threading signal or reduce control
granularity (`FRAME_SKIP=2`) for this section. If adding shaping, keep it generic:
reward "stay near the vertical corridor center / avoid terrain pinch" only if it can
be derived from observations/RAM generally, not a hard-coded L3 y-target.

## Session 4 — frame-skip-2 control-precision experiment

Goal: test the non-destructive control-precision lever without changing the default
`FRAME_SKIP=4` setup or overwriting any best checkpoints.

Code change: `frame_skip` is now an optional env override threaded through:

- `make_env(..., frame_skip=None)` / `make_thunk(..., frame_skip=None)`
- `src.train --frame-skip N`
- `src.play --frame-skip N`
- `tools.capture_frontier_states --frame-skip N`

Default behavior is unchanged (`None` -> `C.FRAME_SKIP`, currently 4).

Before training, deterministic eval under `frame_skip=2` showed the existing hold2
policy was not catastrophically broken by the timing change:

| model / fs | start | steps | max_x | read |
|---|---|---:|---:|---|
| hold2 / fs=2 | `l3_wall` | 241 | 165 | same real-time wall, slight x bump |
| pinch-drill / fs=2 | `l3_wall` | 241 | 158 | no improvement |

Created an isolated curriculum directory `states/l3_pinch_fskip2/` containing copies
of `l3_wall.state` and the reproducible pinch x-threshold states. Then trained:

```
python -m src.train \
  --resume checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip \
  --run-name l3-pinch-fskip2 --timesteps 200000 --ent-coef 0.03 \
  --curriculum-glob 'states/l3_pinch_fskip2/*.state' \
  --curriculum-mix 1.0 --frame-skip 2 --save-freq 100000
```

Training result:

- `curr/best_max_x`: 162 -> **172**
- `curr/best_steps`: 235 -> **242**
- stable PPO metrics; no collapse
- final checkpoint: `checkpoints/l3-pinch-fskip2/lifeforce_ppo_final.zip`

Deterministic eval:

| model / fs | start | steps | max_x | mean_x | read |
|---|---|---:|---:|---:|---|
| hold2 / fs=2 | `l3_wall` | 241 | **165** | 58.5 | baseline under finer control |
| fskip2 / fs=2 | `l3_wall` | 238 | 155 | 60.2 | no transfer/pass |
| fskip2 / fs=4 | `l3_wall` | 118 | 169 | 64.0 | slight x bump, still dies |

Video: `videos/l3_pinch_fskip2_death.mp4`
(`steps=238 max_x=155 terminal_x=16 mean_x=60.2` from `l3_wall.state`, frame-skip 2).

**Conclusion:** frame-skip 2 alone is not the missing lever. It improves local timing
slightly in training metrics but does not produce a deterministic pass from the real
wall start. Keep it as an available flag, but do not replace hold2/pinch-drill as the
reference policy.

### Short clean action-pattern probe

Ran a diagnostic probe from pinch states with fixed movement macros, resetting from
clean saved states for each candidate. Partial result before stopping the broad
search:

- From `x100/x120/x140`, simple forced movement can reach `max_x=232`.
- Those runs die quickly (roughly 20-40 agent-steps from the pinch state).
- Therefore "just force forward/right" can cross screen-x but is not a survivable
threading solution and should **not** be turned into a BC demo.

This reinforces the current diagnosis: the remaining problem is a precise
vertical/terrain threading policy at the pinch, not lack of forward intent or raw
control frequency.

## Session 5 — no terrain signal; generic-only follow-up tests

User concern: adding a vertical/terrain-threading reward may overfit to this one
pinch. Before doing that, we tested additional generic levers only:

1. longer mixed pinch curriculum,
2. score de-baiting via a score-scale override,
3. late-pinch-only curriculum with higher entropy.

### Code additions for safer experiments

Added explicit reward override flags to `src.train` and threaded them into worker
envs:

- `--reward-score-scale`
- `--reward-alive`
- `--reward-death`
- `--reward-xpos`
- `--x-front-frac`
- `--reward-xmax`

These are passed through `make_thunk` / `make_env` / `LifeForceWrapper` rather than
mutating `config.py` at runtime, because `SubprocVecEnv` workers may import modules
fresh. Omitted flags preserve the config defaults exactly.

### 5A. Long mixed curriculum from hold2

Run:

```
python -m src.train \
  --resume checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip \
  --run-name l3-pinch-mixed-long --timesteps 1000000 --ent-coef 0.03 \
  --curriculum-glob 'states/l3_pinch_fskip2/*.state' \
  --curriculum-mix 1.0 --save-freq 100000
```

Stopped after the 800k checkpoint because clean eval plateaued.

Deterministic eval from `states/l3_wall.state`:

| checkpoint | steps | max_x | mean_x | score | read |
|---|---:|---:|---:|---:|---|
| hold2 reference | 117 | 155 | 59.6 | 323 | dies at pinch |
| mixed 500k | 118 | 169 | 67.8 | 423 | modest x/score gain, no pass |
| mixed 600k | 119 | 169 | 66.9 | 423 | no pass |
| mixed 700k | **121** | **169** | 67.9 | 423 | best deterministic candidate so far |
| mixed 800k | **121** | **169** | 69.4 | 423 | plateau; no pass |

Stochastic probe from mixed 700k (`n=48`, clean wall starts): best survival was
121 steps; best max_x was 176 but that run died earlier. No hidden stochastic pass.

**Read:** generic PPO curriculum moved deterministic wall-start from `155/117` to
`169/121`, but did not solve the pinch. Keep `checkpoints/l3-pinch-mixed-long/
lifeforce_ppo_700000_steps.zip` or `800000_steps.zip` as a candidate improvement,
but do not promote it as a pass.

### 5B. Score-muted fine-tune

Hypothesis: score reward had again become too large (`reward/score` around 40-50 vs
`reward/alive` around 18-19), so maybe it was baiting the policy at the pinch.

Run:

```
python -m src.train \
  --resume checkpoints/l3-pinch-mixed-long/lifeforce_ppo_800000_steps.zip \
  --run-name l3-pinch-score001 --timesteps 200000 --ent-coef 0.03 \
  --curriculum-glob 'states/l3_pinch_fskip2/*.state' \
  --curriculum-mix 1.0 --reward-score-scale 0.01 --save-freq 100000
```

Stopped after clean 100k eval.

| model | wall steps / max_x | x120 steps / max_x | read |
|---|---:|---:|---|
| mixed 800k | 121 / 169 | 107 / 169 | baseline |
| score001 100k | 121 / 169 | 107 / 162 | tied/worse |

**Read:** score was not the missing blocker here. Muting it did not reveal a
survival-through-pinch policy.

### 5C. Late-pinch-only entropy drill

Created `states/l3_pinch_late/` with copies of `x140`, `x150`, and `maxx`. This
keeps the training target on the hard decision point without adding a terrain/Y
reward.

Run:

```
python -m src.train \
  --resume checkpoints/l3-pinch-mixed-long/lifeforce_ppo_800000_steps.zip \
  --run-name l3-pinch-late-ent05 --timesteps 300000 --ent-coef 0.05 \
  --curriculum-glob 'states/l3_pinch_late/*.state' \
  --curriculum-mix 1.0 --save-freq 100000
```

Stopped after the 200k checkpoint. Training sampled farther x (`best_max_x=190`),
but `best_steps` stayed 90 from late starts.

Deterministic eval:

| model | wall steps / max_x | x140 steps / max_x | x150 steps / max_x | read |
|---|---:|---:|---:|---|
| mixed 800k | 121 / 169 | 89 / 162 | 86 / 155 | baseline |
| late-ent05 100k | 121 / 169 | 89 / 169 | 86 / 162 | pushes deeper locally, no survival gain |
| late-ent05 200k | 117 / 169 | 89 / 169 | 86 / 162 | wall-start regression |

**Read:** late-only curriculum + entropy teaches "push farther into the pinch" but
not the survivable vertical line. This is the same failure mode as before, now
measured more cleanly.

### Current status after Session 5

Best preserved deterministic wall-start candidate:

- `checkpoints/l3-pinch-mixed-long/lifeforce_ppo_700000_steps.zip` or
  `800000_steps.zip`: `steps=121`, `max_x=169`, `score=423`.

Previous reference still untouched:

- `checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip`: `steps=117`, `max_x=155`,
  `score=323`.

The obstacle is still not passed. The generic-only tests improved "go farther" but
not "survive through the thread." At this point, the evidence says the missing
signal is not forward intent, score weighting, curriculum concentration, or
frame-skip. It is collision/clearance competence at the pinch.

Recommended next lever, if we keep avoiding wall-specific hand coding: add a
general observation-derived clearance/terrain signal, not a hard-coded `y` target
or "go up/down at step N." For example, derive a local navigable-corridor or
distance-to-solid proxy from the preprocessed frame/RAM-visible pixels and reward
staying in safer free space while survival remains the arbiter. That is more
general than an L3-specific vertical script, but the generic PPO-only path looks
spent.

## Session 6 — diagnosis: a narrow timed commit-through, not perception/speed

Stepped back from "add a 6th reward" and looked at two upstream hypotheses.

### Perception check (untested in sessions 1–5)
Dumped what the policy actually sees at the pinch (`states/l3_pinch_curriculum/
l3_pinch_maxx`, ship x=155). Full-RGB: danger is **color-coded** (red terrain,
blue/orange flame column, pink enemies = deadly; dark web = safe) and the corridor
is obvious. The **grayscale-128 obs the policy trains on** keeps gross terrain SHAPE
but **destroys color** — deadly terrain, flames, and safe background collapse into
similar mid-gray textures and the ship is camouflaged. So grayscale does handicap
fine danger-discrimination. (Images: scratchpad `perc_l3_pinch_*_{rgb,gray}.png`.)
All 5 prior sessions held the obs fixed and varied only reward/curriculum/control —
all plateaued at the same 169/121, which points upstream of reward.

### But the real blocker is TIMING/decision, not perception or speed (user's call)
Speed trace of greedy (mixed-700k) from `l3_wall` — **speed (RAM 0x80) is constant
2 the whole time**; the user notes the whole game is clearable at 0 speed, so speed
is a non-issue. The x-column is the story:

```
step   0: x=29   ...   step 20: x=169  (reaches the FRONT, at the right moment, alone)
steps 21–30: hovers x=162→148, starts pulling back
steps 32–80: full RETREAT x=127→99→57→15
steps 80–112: camps at x=15 and dies
```

**The ship gets to the gap at the right time and then retreats** — the stay-back
prior (correct elsewhere: gauntlet #1, closing walls) overrides at the one spot it
shouldn't. The survivable maneuver is a **narrow, precisely-timed commit-through the
gap during ≈ steps 17–24 (x≈148–169, y≈143)**. It is NOT perception, speed, or
"go faster" — it's *commit forward through the window instead of retreating*.

### Why 5 sessions of RL couldn't fix it
The survivable commit is a narrow, precisely-timed maneuver AND the policy is
*actively retreating* at exactly that moment, so on-policy exploration almost never
samples "hold forward through the window," and the stay-back prior is too entrenched
for entropy alone. **Textbook case for demonstration, not discovery.**

### Decision (next session) — self-imitation from a REPRODUCIBLE demo
Plan: produce **one clean demo of the timed commit-through → BC it (`self_imitation.py`)
→ resume PPO** (survival reward reinforces it; user is confident the maneuver
survives). Key: the demo must come from a **saved state** — reset-to-saved-state +
replay is deterministic (the determinism probe showed reloads reproduce 2/2); the
earlier broken Go-Explore demo (`l3_pinch.npz`, archived) was non-reproducible only
because of its greedy "advance-to-near-failure" phase, NOT reload.

**Chosen demo source: script/search it (targeted, NOT broad Go-Explore).** From a
saved approach state (e.g. `states/l3_pinch_curriculum/l3_pinch_x120.state`, or a
state captured ~step 14–16 just before the window), force a decisive RIGHT (+ the
right Y) hold through the step-17–24 window and tune until one candidate survives
well past x=155 with a clean greedy continuation. That action sequence (obs+actions
from the saved state) becomes the BC demo. Then `self_imitation` → PPO.
- Do NOT use the broad existence-search (we agree the path exists; the search's only
  job is to PRODUCE the reproducible demo).
- Reference policy to demo/BC from: `checkpoints/l3-pinch-mixed-long/
  lifeforce_ppo_700000_steps.zip` (best deterministic wall-start: 121 / max_x 169 /
  score 423) or `checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip`.

## Session 7 — regression check + structured segment search

### Full-level regression check (finally run; deterministic, 1 ep from true level start)

| checkpoint | steps | score | max_x | mean_x |
|---|---:|---:|---:|---:|
| 2M baseline (`lv1-front-speed/2000000`) | **891** | 190 | 226 | 119 |
| hold2 (`l3-wall-hold2/final`) | 566 | 190 | 232 | 189 |
| mixed-700k (`l3-pinch-mixed-long/700000`) | **75** | 9 | 231 | 190 |

**The pinch specialists regressed hard on the full level.** mixed-700k front-rushes
from the level start (mean_x 190) and dies at 75 steps — the front-shaping +
`--curriculum-mix 1.0` (never seeing level starts) re-broke the early level,
exactly the gauntlet-#1 failure the config comments warned about. Videos:
`videos/regress_{2M-baseline,hold2,mixed-700k}.mp4`.

**Consequence for the plan:** mixed-700k stays the base for the *pinch* demo/BC
work (its local 121/169 approach is what matters at the wall), but after the pinch
is solved a **consolidation phase is mandatory**: PPO with the full curriculum
(level start included, mix well below 1.0), judged by full-level steps vs the 2M
baseline's 891. The L3 branch is currently a specialist, not a level-1 policy.

### Structured segment search (`tools/segment_search.py`, new)

Session 6 decision implemented: exhaustive sweep of 3-segment (move × duration)
plans — {R, UP+R, DOWN+R, UP, DOWN} × {4,8,12,16} agent-steps = 8000 plans — from
a FRESH reload of `l3_pinch_x120.state` per candidate (no advance phase; the
non-reproducibility source is designed out), scored by greedy-handoff survival,
winners verified by 2 independent reloads before any demo/state is written.
Determinism probe passes; greedy baseline from x120 = 107 steps / max_x 169.
Smoke finding: single UP+R×8 reaches max_x=205 (past the pinch) but handoff dies
at 26 steps — forward-only crosses x but isn't survivable (consistent with S4).

### Exhaustive sweep result: dry, but decisive diagnosis

All 8000 plans: none beat the 107-step baseline; top plans reach max_x=211 with a
clean handoff yet die at EXACTLY 107 — identical to greedy alone. Trace of the best
plan (`videos/l3_plan_trace.mp4`): script puts the ship at x=204–211 y=143 through
the gap zone (steps 10–20), then greedy RETREATS 211→15 by step 65, camps, and dies
at 107 on schedule. **Any ≤48-step script is erased by the retreat prior; the
handoff must happen after the dangerous terrain has scrolled past (~step 107+).**

### Beam search (v1: fixed-move vocab)

Added `--beam` mode: grow the script segment-by-segment, prune branches that die
in-script, rank by total survival (script + greedy continuation), verify winners
2× from fresh reloads. Result: **round 7–8 breakthrough — 198 total steps (+91 vs
107), script 76 steps, greedy continuation 122** — first time greedy survives past
the camping death; new death at ~198 is the next obstacle downstream. Rounds 9–13
declined (147): the 198 branch died out of the beam because fixed-move segments
can't out-play greedy in the stretch greedy already handles (steps ~76–198).
Killed at round 13 (deterministic: `--rounds 9`, seed 0, no `--greedy-move`
reproduces the 198 line exactly if we ever want to bank it).

### Beam v2: GREEDY pseudo-move — FAILED (negative result, keep for the record)

Added `--greedy-move`: a segment that FOLLOWS the deterministic policy for d steps
(recorded, recomputed per replay — reproducible). Hypothesis: the beam rides the
policy where it's already right and scripts only the corrections. Acceptance: greedy
continuation ≥150 steps and gain ≥30, verified 2×.

Result (width 8, seed 0, log `logs/l3_beamg_v2.log`): best total pinned at EXACTLY
107 (+0) for all 12 rounds, then round 13 exhausted the beam — every extension died
in-script. **Why it failed:** GREEDY is the safest move in-script, so greedy-mimicking
branches dominate the survival ranking; but pure greedy IS the step-107 camping death,
so every retained branch funnels into it, and with all survivors tied at 107 steps the
beam's tie-breaking is random — the corrective scripted prefixes that produced v1's
198 line get crowded out and never return. Survivor count collapsed 137→62→18→0 over
rounds 10–13. Lesson: don't put the baseline policy in the vocab when the baseline is
the failure mode being searched around.

### Banking v1's 198 line (`demos/l3_pinch_beam198.npz`) — DONE

Re-ran the v1 config (fixed-move vocab, `--beam 8 --rounds 9 --seed 0`, no
`--greedy-move`) with `--name l3_pinch_beam198`. Reproduced: round 9 best = **198
total steps (+91), max_x 232**, script `DOWNx4 > Rx4 > Rx8 > UPx4 > DOWNx4 > UP+Rx16
> Rx16 > DOWN+Rx16 > Rx4` (76 steps, continuation 122), REPRODUCED 2/2. Saved
`demos/l3_pinch_beam198.npz` + handoff `states/l3_pinch_beam198.state`.
Log: `logs/l3_beam198_bank.log`.

### BC (`tools/self_imitation.py`) — maneuver transfers, but open-loop

10 epochs on the 76-step demo, loss 3.38 → 0.14 → `checkpoints/l3-bc/lifeforce_ppo_bc.zip`.
Greedy eval from x120 (empty-plan rollout):

| policy | steps | max_x | read |
|---|---:|---:|---|
| base mixed-700k | 107 | 169 | camps, dies at 107, never crosses |
| BC | 66 | **232** | charges THROUGH the pinch like the demo, dies mid-maneuver |

BC flipped the behavior from "retreat and camp" to "attempt the demo maneuver" but
can't execute it closed-loop (compounding error) — exactly the gap PPO's survival
reward is supposed to close.

### PPO robustify (`l3-bc-ppo`) — PINCH SOLVED (as a specialist)

```
python -m src.train --resume checkpoints/l3-bc/lifeforce_ppo_bc.zip \
  --run-name l3-bc-ppo --timesteps 500000 --ent-coef 0.05 \
  --curriculum-glob 'states/l3_bc_curriculum/*.state' --curriculum-mix 1.0 --save-freq 50000
```

Curriculum = {x100, x120, **beam198 handoff**} (handoff = train from PAST the pinch;
go-explore: train from the frontier). 500k steps, ~11 min on mps. Deterministic
greedy eval (empty-plan rollout), all checkpoints:

| ckpt | x120 steps/max_x | wall steps/max_x |
|---|---|---|
| base mixed-700k | 107 / 169 | 121 / 169 |
| scripted 198 line | 198 / 232 | — |
| 50k | 104 / 232 | 114 / 232 |
| 200k | 217 / 232 | 229 / 232 |
| 400k | 264 / 232 | 240 / 232 |
| **final** | **318 / 232** | **276 / 232** |

**Every checkpoint crosses the pinch closed-loop from 50k on; final survives 318
steps from x120 — +120 past the scripted line.** The BC seed + survival reward
closed the compounding-error gap completely. Note max_x saturates at exactly 232
everywhere (same value hold2 hit on the full level) — 232 is the next barrier (or an
x-counter cap); gains past ~200k steps are survival AT 232, not progress past it.

Full-level regression (deterministic, from true level start): **75 steps** — same
as mixed-700k. The specialist front-rushes (mean_x 196) and dies early. Expected;
triggers the consolidation phase.

### Consolidation (`l3-consolidate`) — level start recovered AND pinch retained

```
python -m src.train --resume checkpoints/l3-bc-ppo/lifeforce_ppo_final.zip \
  --run-name l3-consolidate --timesteps 500000 --ent-coef 0.03 \
  --curriculum-glob 'states/l3_bc_curriculum/*.state' --curriculum-mix 0.5 --save-freq 50000
```

Mix 0.5 = half the episodes from the TRUE level start, half from the pinch states.
Judged on BOTH bars. Deterministic sweep (corrected — see eval-gotcha below):

| ckpt | full-level steps/max_x | x120 steps/max_x |
|---|---|---|
| pre (l3-bc-ppo final) | 75 / 232 | 318 / 232 |
| 2M baseline (reference) | 891 / **226** | — (camps at 107/169) |
| 100k | 526 / 233 | 292 / 232 |
| 400k | 775 / 233 | 416 / 233 |
| **final** | **775 / 233** | **428 / 232** |

Level-start survival 75 → **775** while x120 stayed 292–428 (bar 198) at every
checkpoint. **The consolidated policy crosses the pinch from a cold level start**
(max_x 233 vs the baseline's 226 — the baseline never crossed; its 891 steps were
wall-camping). Remaining gap: raw survival 775 vs 891.

**Eval gotcha (burned once):** `tools.segment_search.rollout` sets
`env.unwrapped.initial_state` — after any state-load, `env.reset()` NO LONGER
returns to the level start. Save `env.unwrapped.initial_state` before the first
load and restore it before level-start evals, or the "full level" numbers are
silently x120 numbers (symptom: both columns identical).

### Consolidation round 2 (`l3-consolidate2`) — converged; ACCEPTED as new best

Same recipe from `l3-consolidate/final`, `--curriculum-mix 0.3` (70% level starts).
Full-level plateaued at ~805–812 across six checkpoints (peak 812 @300k; final
**811 / max_x 233**, x120 **434**). 891 not reached, but accepted per the
progress-beats-camping rule: baseline's 891 was wall-camping at x≤226; this policy
crosses the pinch cold. x120 dipped to 190 at the 200k checkpoint only (transient —
sweep checkpoints, never trust just the final).

**NEW BEST OUR-LINEAGE FULL-LEVEL POLICY:
`checkpoints/l3-consolidate2/lifeforce_ppo_final.zip`**
(full level 811 steps / max_x 233 / score 190; x120 434 steps).

### CORRECTION — max_x semantics (this reframes the "232 barrier")

`x_pos` (RAM 0x350) is the ship's SCREEN position, hard range 15..232
(`C.X_POS_MIN/MAX`; can blip to 234 at scroll transitions). Life Force auto-scrolls
(clock at RAM 0x2F advances ~0.25/frame regardless of input), so **steps survived
IS level progress; max_x only measures how far front on screen the ship ventured.**
Therefore:

- There is NO "x=232/233 barrier" — that's the screen edge. The next obstacle is
  the terrain event at scroll-time ~step 811 (from level start).
- The consolidate2 acceptance argument "max_x 233 > 226 = more progress than the
  camping baseline" was WRONG — screen position isn't progress. On the honest
  metric the 2M baseline (891 steps) still survives ~80 steps deeper than ours
  (811). **The 891 full-level bar stands, currently unmet.** consolidate2/final
  remains the best policy of the BC lineage and keeps the x120-start skill
  (107 → 434 steps), but it dies at a step-811 event the baseline survived —
  likely BECAUSE of its front-venturing habit; a survivable positioning exists.
- Earlier "crosses the pinch (max_x 232)" phrasing = "ventures to the front of the
  screen through the pinch section"; the real signal was always the step gains.

### Trajectory anatomy + next frontier (step 404)

`capture_frontier_states` on the policy's full-level run: front-most excursion at
step 404, then it retreats to the rear (screen x=16) and dies at step 811 (mean_x
136). Captured `states/l3_b232_maxx.state` (step 404) as the search start — ~400
steps of lead-in before the step-811 death.

### Go-explore iteration 2 — two beams, one winner

Two parallel beam searches at the step-811 death (proven recipe: fixed-move vocab,
`--beam 8 --seed 0`, NO --greedy-move):

- **From step 404 (`l3_b232_beam`): killed at round 4.** The death sits ~400 steps
  out but `--script-cap` is 200 — the script can only steer the handoff, and every
  branch funneled to the same death (+1 after 4 rounds, ~11 min/round). Lesson:
  **start the beam ~120 steps before the death** (like the pinch search), not at
  the front-most excursion.
- **From step 691 (`l3_b811_beam` = `l3_b232_bd120.state`, 120 before death):
  ACCEPTED at round 11.** Baseline 119; rounds 1–10 pinned at ±2 (short 4-step
  wiggle segments), then round 10 grew forward segments and round 11 broke through:
  script `UPx4 > DOWNx4 > UPx4 > UPx4 > DOWNx4 > Rx4 > Rx4 > DOWN+Rx12 > Rx16 >
  UPx16 > UP+Rx8` (80 steps) → **400 total steps (+281), continuation 320,
  REPRODUCED 2/2**. From level start that's ~step 1091 — 200 past the 891 bar.
  Demo `demos/l3_b811_beam.npz`, handoff `states/l3_b811_beam.state`.

### BC round 2 — 10 epochs too weak, 20 works

BC of the 80-step demo into `l3-consolidate2/final`: at 10 epochs (loss 0.20)
greedy-from-bd120 stayed 120 (no transfer); at **20 epochs** (loss 0.009) it
reaches **139 — past the step-119 death**. `checkpoints/l3-bc2/lifeforce_ppo_bc20.zip`.
Side effects as last time (x120 434→250, full level 811→65) — consolidation's job.

### PPO robustify round 2 (`l3-bc2-ppo`, running)

```
python -m src.train --resume checkpoints/l3-bc2/lifeforce_ppo_bc20.zip \
  --run-name l3-bc2-ppo --timesteps 500000 --ent-coef 0.05 \
  --curriculum-glob 'states/l3_b811_curriculum/*.state' --curriculum-mix 1.0 --save-freq 50000
```

Curriculum = {bd120, b811 handoff, x120}. Sweep: bd120 up to **532** (350k–500k),
x120 collapsed to 23 mid-run then recovered to 434 by 500k, full-level broken (~65,
expected). Consolidation base = 500k.

### Consolidation round 3 (`l3-consolidate3`) — **891 BAR BROKEN**

```
python -m src.train --resume checkpoints/l3-bc2-ppo/lifeforce_ppo_500000_steps.zip \
  --run-name l3-consolidate3 --timesteps 500000 --ent-coef 0.03 \
  --curriculum-glob 'states/l3_b811_curriculum/*.state' --curriculum-mix 0.3 --save-freq 50000
```

| ckpt | full steps | score | bd120 | x120 |
|---|---:|---:|---:|---:|
| 2M baseline | 891 | 190 | — | — |
| 250k | 1086 | 696 | 390 | 462 |
| 350k | 1065 | 690 | 514 | 462 |
| **400k** | **1111** | **699** | 374 | 432 |
| 450k–final | ~530 | 90 | ~400 | ~300 |

**NEW BEST POLICY: `checkpoints/l3-consolidate3/lifeforce_ppo_400000_steps.zip` —
full level 1111 steps / score 699** (+220 steps, 3.7× score vs the 2M baseline),
with both drilled skills retained. Checkpoints after 400k washed out to ~530/90:
the sweep, not the final checkpoint, is what finds the winner. Two full go-explore
iterations (pinch, step-811) both banked into one policy.

### Go-explore iteration 3 (step-1111 death) — trivial dodge found instantly

Frontier captured from the 1111 run (front-most at step 738; death at 1111,
terminal screen-x 16). Beam from `l3_b1111_bd120.state` (step 991) accepted in
ROUND 1: **a single `UPx4` → +94** (214 total, cont 210, REPRODUCED 2/2) →
`demos/l3_b1111_beam.npz` (4 steps) + `states/l3_b1111_beam.state`. The hazard is
a near-miss mispositioning, and the terrain past 1111 is easy for ≥200 steps.
Re-ran the beam with `--accept-cont 99999` (`l3_b1111_deep`) to bank the deepest
script 13 rounds can find: global best from round 8 verified 2/2 — 96-step script,
**218 total (+98)** → `demos/l3_b1111_deep.npz`. Only +4 over the UPx4 dodge.

### BC on a 4-step demo — FAILED (lesson: match the tool to the maneuver size)

20 epochs on the 4-sample UPx4 demo wrecked the policy locally (bd120 120 → 15):
gradient-hammering 4 frames bends everything nearby. Checkpoint deleted.
**Rule: BC for long precise maneuvers (≥~30-step demos); tiny dodges → plain PPO
drilling, exploration finds them.**

### Drill without BC (`l3-drill1111`) — dodge learned, full level jumps to 1245

PPO directly from consolidate3/400k, mix 1.0, curriculum {b1111_bd120, b1111
handoff, x120, b232_bd120}. Sweep: dodge learned (b1111 120 → 263–300), b811 up to
624, x120 stable ~455 — and full-level jumped to **1245–1251** at 50k/450k/500k
even with zero level-start episodes (the local fix propagates through the
deterministic trajectory). Final washed out (504). Base for consolidation: 500k.

### Consolidation round 4 (`l3-consolidate4`) — stable at ~1255, NEW FLAGSHIP

Mix 0.3 from drill/500k. Nearly EVERY checkpoint holds full-level ~1243–1268 with
all skills intact (b1111 ~250–300, b811 ~550–600, x120 ~470) — the behavior is
robust now, not knife-edge. **FLAGSHIP:
`checkpoints/l3-consolidate4/lifeforce_ppo_400000_steps.zip` — full level 1261
steps / score 708** (session start: 891 / 190).

### Go-explore iterations 4–5 (steps 1261, 1398) — routine cycles

Same recipe, both landed: iteration 4 banked `Rx12` (+59) at step-1261 → drill
(no BC) → consolidate5 → **flagship 1398/811** (`l3-consolidate5/50000`).
Iteration 5 banked a 20-step dodge (+96) at step-1398 → drill → consolidate6 →
**flagship 1736/1314** (`l3-consolidate6/final`; score jump 811→1314 = a whole
new scoring section). Drills consistently overshoot their beam's local gain
(the local fix propagates through the deterministic trajectory), and the best
checkpoint is consistently mid-run — sweep everything.

### Go-explore iteration 6 (step-1736 gauntlet) — search SOLVED, robustification BLOCKED

The hardest section yet, and the first cycle the pipeline could not close:

- Coarse beam (bd120, durations 4/8/12/16): pinned at +14 after 13 rounds — FAIL.
- Fine beam (bd80, durations 2/4/8, 20 rounds): 68-step script, +57, verified.
- Extended (40 rounds): 178-step script, +125. Extended again (60 rounds,
  script-cap 400): **267-step script, +231, verified 2/2 → ≈ step 1964 from
  level start** (`demos/l3_b1736_fine3.npz`). Deterministic seed = each
  extension re-treads the previous rounds for free.
- **BC 20 epochs: best transfer of the session — greedy 389 steps from bd80**
  (beats the script's 308). But BC wrecked everything else (b811 16, x120 21).
- **Every robustification attempt eroded the corridor**: drill mix 1.0
  (491→90), consolidate mix 0.3 ent 0.03 (peak full 1629 < bar, corridor ~90),
  consolidate from BC5 ent 0.01 (mode collapse: one 485-step score-0 line).

**Mechanism:** the 267-step corridor has near-zero action tolerance. Stochastic
training rollouts die in it constantly, so PPO learns avoidance even though the
deterministic policy threads it; BC implants it at the cost of everything else;
too-low entropy can't rebuild what BC broke. Implant vs erosion is currently a
strict trade-off. Rescue ideas ranked in RESUME.md (demo-ensemble BC for
tolerance, KL-anchored/interleaved self-imitation, scroll-depth reward shaping,
entropy annealing).

**Session 7 final flagship: `checkpoints/l3-consolidate6/lifeforce_ppo_final.zip`
— 1736 steps / score 1314** (vs 891/190 at session start; +95% survival, 6.9×
score; five hazards banked and consolidated; the sixth found-but-not-yet-learned).

## Artifacts produced this session

- Demos: `demos/l3_bridge.npz`, `l3_bridge_m40/60/80.npz`, `l3_bridge_long.npz`
  (144-step maneuver), `l3_pass.npz`. **All "camp and die later" — none is a clean
  obstacle pass.**
- Frontier states saved (60-step runs only): `states/l3_bridge.state`,
  `l3_bridge_m40/60/80.state` — these put greedy back at the same wall, **not
  past it**.
- Video: `videos/l3_wall_death.mp4`.
- Search logs: scratchpad `explore_l3*.log`.
