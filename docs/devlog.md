# Devlog — decisions, lessons, current state

The experiential knowledge that isn't in the code or the other docs: *why* things
are the way they are, what we tried, and where the project stands. Read this to
pick up the project cold. (Reference: [`ram_map.md`](ram_map.md),
[`macos_arm64_build.md`](macos_arm64_build.md); usage: [`../README.md`](../README.md).)

---

## Current state & next steps (read this first)

**Goal:** clear Level 1. **Status:** the agent reliably reaches a mid-stage
terrain **"gauntlet"** (organic walls that narrow the passage) but **cannot
thread it** — training plateaus there. It does *not* yet clear the level.

**The wall, concretely:** from the level start the agent dies at ~**615 steps**
at internal score ~**262** (HUD `2620`). Curriculum drilling pushed the best to
~**280** but that was mostly extra kills at the same chokepoint, not getting
past it (confirmed by watching a replay).

**Levers to get past the gauntlet, in priority order (cheapest first):**
1. **Loadout fix (in progress).** The agent poured every capsule into **Speed**
   (`speed=7`) → the ship overshoots and can't make fine corrections in the gap.
   Prioritize **Missile > Option > Force Field**; for speed, we **hard-cap** it at
   `MAX_SPEED=2` via **action masking** (the Discretizer refuses to activate the
   power-up on the speed slot once speed ≥ cap). *Reward penalties failed first:*
   flat `-1.0`, then a threshold `-5.0`/over-level — the agent over-sped **anyway**
   both times (`reward/powerup` started positive then declined as it learned to
   over-speed past the penalty). Reason: speed is genuinely net-positive early
   (dodge → survive → score, hundreds of reward) and the gauntlet-overshoot cost
   is ~600 steps away → γ-discounted to ~0, so no affordable penalty beats it. The
   hard cap sidesteps the arms race and cleanly **tests the hypothesis**: if a
   ≤2-speed agent threads the gauntlet, speed was the blocker; if not, it wasn't.
   Run fresh (no speed habit to fight).
2. **Earlier curriculum capture.** The captured wall state was ~30 steps before
   death with the ship already cornered at `x=14` (far-left) — likely an
   *unwinnable* position. Re-capture with a bigger lead-in (`--before-death 120`)
   so the agent drills the *approach*, not the death frame. Do this once the
   agent reaches the wall *slow* (low speed).
3. **Exploration** (`--ent-coef 0.03`) — untested as a deliberate lever; may help
   escape the stuck strategy at the wall.
4. **Perception** (`FRAME_SIZE` 84→128) — the gap is a few pixels at 84×84
   grayscale; the agent may literally not resolve it. Expensive (fresh train,
   ~9× cost at 256, so try 128). **Last resort**, only after 1–3 fail.

**Immediate recommended action:** fresh train with the no-speed reward (curriculum
off / `states/` empty) so it learns the correct loadout from scratch; watch
`reward/powerup` go **positive** (right loadout) and `lifeforce/best_score` for
movement past 280.

---

## Key empirical findings

- **Clearing is hard.** The bundled `LifeForce-Nes-v0` integration ships a 10M-step
  PPO benchmark tagged **`bad-end`** — even 10M steps didn't cleanly clear. Set
  expectations accordingly.
- **Score is displayed ×10.** Internal score counter = HUD/10 (internal 262 = HUD
  `2620`). `reward/score` ≈ internal_score_delta × 10.
- **The gauntlet is deterministic** (fixed terrain) → memorizable with focused
  practice, which is why save-state curriculum is the right tool.
- **Throughput: ~278 fps on CPU, env-bound.** More `N_ENVS` barely helps (the
  bottleneck is emulation/IPC, not the tiny NatureCNN). **MPS is ~25% slower**
  than CPU (CPU↔GPU transfer overhead with no compute win). Train on CPU.
- **macOS sleeps pause training** — use `caffeinate -is` for overnight runs.

---

## Training journey & lessons

1. **Reward design (survival vs camping).** First reward (score + flat per-step
   alive bonus) caused **camping**: `alive` rose while `score` fell. Fix: make
   survival #1 via **ending the episode on death** (death forfeits all future
   reward) instead of a big idle bonus, so the agent stays alive *to* score.
   Score is the main positive → active play. Lesson: a flat alive bonus pays the
   agent to idle; episode-termination is the better survival incentive.
2. **Policy collapse (~240k).** Training improved then *crashed permanently*
   (`approx_kl` spiking to 0.03–0.04, `clip_fraction` ~0.45). Fix: **`target_kl`
   = 0.02** (early-stop oversized updates) + **linear LR annealing**. Stopped the
   permanent collapse.
3. **Large oscillation (critic overwhelmed).** Stable but reward swung wildly
   (`value_loss` 500–1200, `explained_variance` crashing to 0.15). Cause: returns
   ~hundreds → value targets too large/noisy. Fix: **reward normalization**
   (`VecNormalize(norm_reward=True)`). Result: smooth monotonic climb,
   `value_loss` → tiny, `explained_variance` → 0.95+. **This was the big unlock.**
4. **Action space.** Started `Discrete`; switched to **`MultiDiscrete([9,2])`**
   (movement × activate-power-up) because the old set forced a choice between
   moving and activating, so the agent under-used weapons (esp. under
   `--deterministic`, which suppresses the rare activate action).
5. **Plateau → curriculum.** Smooth training plateaus at the gauntlet. Built
   **save-state curriculum** (drop a `.state` in `states/`, auto-mixed into
   training). It nudged best_score 262→280 then re-plateaued.
6. **Cornered-capture lesson.** The plateau was partly because we were drilling an
   *already-lost* state (ship cornered at `x=14`). Capture *earlier*.
7. **Loadout lesson.** The agent over-invested in Speed, which *hurts* a precision
   navigation task. Penalize speed (current frontier).
8. **Measurement lesson.** Under curriculum, `reward/*` averages are **diluted**
   by short hard wall-start episodes — they look bad even when progressing. Judge
   progress by **`lifeforce/best_score`** (absolute score, which the save state
   preserves), *not* the reward averages.

---

## Design decisions & rationale

- **stable-retro (not a turnkey package).** The whole point: do RL on a game with
  *no* package — build from source, BYO-ROM, extend the integration ourselves.
- **84×84 grayscale start.** The proven RL default (DQN/NatureCNN); cheap; it was
  sufficient for everything except possibly the gauntlet. Scale resolution only
  where evidence demands (the wall), and by the smallest step (128 before 256).
- **Reward = delta-based.** The agent's reward rewards *changes* (score gained,
  upgrades acquired). Measurement uses *absolute* score (`info["score"]`), which
  the save state preserves — so `best_score` tracks true progress.
- **Power-up shaping rewards state *increases*** → upgrade caps self-enforce (a
  maxed upgrade can't rise → no reward → no wasted-capsule incentive). No need to
  hardcode caps.
- **Per-run checkpoint folders** (`checkpoints/<run-name>/`) because SB3 resets the
  step counter on resume and was overwriting prior runs' checkpoints.

---

## Open questions / TODO

- **Stage-clear detector unconfirmed.** We never reached Stage 2, so the
  "cleared" RAM signal (`0x23` stage-num / `0x40` vertical-flag) is still a
  hypothesis. The env auto-captures the first Stage-2 transition to `ram_dumps/`
  for confirmation — but it hasn't fired (never gotten past the wall).
- **Phase 2: auto-curriculum.** Designed, not built. A callback that detects
  "stuck" (no new `best_score` in ~N steps), auto-captures the current death point
  to `states/` (envs pick it up live via re-glob), and drills it — automating the
  plateau→capture→drill loop. Build only after manual curriculum demonstrably
  breaks a wall.
- **Does the loadout fix (no speed) actually break the gauntlet?** The current
  open experiment.
