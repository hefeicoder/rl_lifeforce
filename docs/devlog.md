# Devlog — decisions, lessons, current state

The experiential knowledge that isn't in the code or the other docs: *why* things
are the way they are, what we tried, and where the project stands. Read this to
pick up the project cold. (Reference: [`ram_map.md`](ram_map.md),
[`macos_arm64_build.md`](macos_arm64_build.md); usage: [`../README.md`](../README.md).)

---

## Current state & next steps (read this first)

**Goal:** clear Level 1. **Status:** the **first** mid-stage "gauntlet" (narrowing
terrain) is **SOLVED** — the agent threads it once it stops hugging the front edge
(see the positional cap below). `best_score` broke its long ~280 plateau and now
sits at **~380**. It still does *not* clear the level: it's stuck at a **new** wall,
a **branching fork** further in.

**History of the first wall (solved):** the agent plateaued at internal score ~**280**
(HUD `2800`) for every run — ruled out loadout, curriculum-practice, and exploration
(all below). The real cause, found by *watching a replay*: the ship **hugged the
front/right edge** with no time to react to terrain scrolling in. Hard-capping its
forward position (mask RIGHT past `X_SAFE_FRONT`) broke it → **280 → 380**, confirmed
both on resume (`lv1-stayback`) and a **fresh from-scratch 16-env run**
(`lv1-speed-improve`, 1M steps / 17 min — also the sanity check that the new MPS +
N_ENVS=16 defaults preserve learning).

**The current wall — the fork (380):** after the first gauntlet the path **branches**;
the **upper channel is correct, the lower is a dead end** (user observation, replay).
The agent can't reliably pick upper — `recent_best_score` is **bimodal** (~270 when
it takes the dead end, ~380 when it goes up). It's a deterministic branch with a
**delayed-trap credit-assignment** problem: the dead-end death comes ~100+ steps
after the choice, γ-discounted to a weak signal, and from a full-level start the
agent reaches the fork on few episodes. **Next: drill the fork with save-state
curriculum** (capture just before the dead-end death → every episode starts at the
decision → strong, concentrated signal). Fallbacks if drilling fails: raise γ
(0.99→0.997) to propagate the distant signal; recurrent policy only if the fork is
truly visually ambiguous (suspected not — the scenery is a distinctive landmark).

**Levers to get past the gauntlet, in priority order (cheapest first):**
1. **Loadout fix — DONE; fixed the loadout but did NOT break the wall.** The agent
   poured every capsule into **Speed** (`speed=7`) → the ship overshoots and can't
   make fine corrections in the gap. We **hard-cap** speed at `MAX_SPEED=2` via
   **action masking** (the Discretizer refuses to activate the power-up on the
   speed slot once speed ≥ cap). *Reward penalties failed first:* flat `-1.0`, then
   a threshold `-5.0`/over-level — the agent over-sped **anyway** both times
   (`reward/powerup` started positive then declined as it learned to over-speed
   past the penalty). Reason: speed is net-positive early (dodge → survive → score,
   hundreds of reward) and the gauntlet-overshoot cost is ~600 steps away →
   γ-discounted to ~0, so no affordable penalty beats it. **Result of the fresh
   `lv1-speedcap_2` run (~800k steps):** the cap worked exactly as designed —
   `reward/powerup` flipped from **−4.25 → +11.29** (capsules now go to
   Missile/Option/Force Field) and in-segment `reward/score` **doubled** (~1078 →
   ~2284). **But `lifeforce/best_score` stayed pinned at ~280** (same plateau as
   every prior run) and `clear_rate` stayed 0. So the doubled score was *more
   scoring at the same chokepoint before dying*, not getting further.
   **Verdict: speed was NOT the blocker** — a real behavioral bug, now fixed (keep
   the cap), but the gauntlet has a second, independent cause. Loadout is ruled out.
2. **Earlier curriculum capture (NOW the frontier).** Previously blocked: the
   captured wall state was ~30 steps before death with the ship already cornered at
   `x=14` (far-left) from *over-speeding* — an unwinnable frame. With the speed cap,
   the agent now arrives **slow and controlled**, so a capture of the *approach*
   (`--before-death 120`) is finally meaningful. Drill it off the `lv1-speedcap_2`
   checkpoint and watch `best_score` for movement past 280.
3. **Exploration — DONE; ruled out.** Resumed the 900k policy with curriculum +
   raised entropy: `lv1-drill-explore_1` (`--ent-coef 0.03`) and `_2` (`0.05`).
   `train/entropy_loss` confirmed the bump took effect (−2.4 → −2.63 at 0.03, −2.71
   at 0.05 — more entropy), and training stayed healthy (`explained_variance` ~0.90,
   `approx_kl` ~0.003, `reward/powerup` positive). **Still exactly 280, clear_rate
   0.** So more exploration doesn't crack it either. **All three cheap levers
   (loadout, curriculum/practice, exploration) are now exhausted — the blocker is
   structural, not a reward/training-config problem.**

   **Diagnostic before spending a fresh train:** watch the death at the gauntlet
   (`python -m src.play --model <ckpt> --from-state states/<wall>.state
   --deterministic`) and characterize *how* it dies — that picks lever 4 vs 5:
4. **Perception** (`FRAME_SIZE` 84→128) — if it crashes into / clips a wall it
   "should" see. At 84×84 the gap is a few pixels; downscaling from NES's 256-wide
   may erase it. Fresh train, ~2.3× input area at 128 (the 9× figure was for 256).
5. **Control precision** (`FRAME_SKIP` 4→2) — if it clearly *sees* the gap but
   mistimes (overshoots, can't correct fast enough). At skip-4 the agent acts only
   15×/sec; threading a narrowing gap may need finer timing than that can express.
   Fresh train; episodes get longer (more decisions/sec) so slower wall-clock.

**Positional cap — CONFIRMED, this is what broke the first wall.** Watching a replay
of the stuck policy: the ship **hugged the leading (front/right) edge**, so terrain
and enemies scrolling in from the front gave it no time to react — a textbook shmup
death. Fix: **mask the RIGHT button once x_pos ≥ `X_SAFE_FRONT` (=100)** so the ship
physically can't advance past the back ~40% (it can still retreat, hover, move
vertically). Same hard-mask pattern as the speed cap, same reasoning (a positional
penalty would fight an arms race). **Result: `best_score` 280 → 380** — confirmed on
a resume (`lv1-stayback`) *and* a fresh from-scratch run (`lv1-speed-improve`). This
was the structural cause all along; the perception/frame-skip levers below were never
needed for the first wall (keep them in reserve for the fork only if drilling fails).
Note it's an action-only change, so it applies on resume *and* fresh with no obs
change. Tunable: `X_SAFE_FRONT` lower (~70) for stricter back-25% if a later section
needs it.

**Immediate recommended action:** drill the **fork** (the current 380 wall). Capture
a state ~120 steps before the dead-end death off the best checkpoint, then resume —
every episode starts at the decision so the delayed-trap signal becomes strong. Watch
`lifeforce/best_score` for movement past 380. This also *validates the capture→drill
loop* before we automate it (auto-curriculum, phase 2).

---

## Key empirical findings

- **Clearing is hard.** The bundled `LifeForce-Nes-v0` integration ships a 10M-step
  PPO benchmark tagged **`bad-end`** — even 10M steps didn't cleanly clear. Set
  expectations accordingly.
- **Score is displayed ×10.** Internal score counter = HUD/10 (internal 262 = HUD
  `2620`). `reward/score` ≈ internal_score_delta × 10.
- **The gauntlet is deterministic** (fixed terrain) → memorizable with focused
  practice, which is why save-state curriculum is the right tool.
- **Throughput: the bottleneck moves twice — full story (was wrong before).** The
  old "env-bound, train on CPU, MPS 25% slower" claim was backwards. Profiled with
  `tools/bench.py` on an M1 Max (10-core); combined result is **~5.5x over the
  original CPU/8-env setup (250 → 1378 fps)**. How:
  1. **Env stepping is NOT the bottleneck.** It runs ~2700 agent-sps (~10800
     emulated fps) — ~10x the training rate. (An env-only N_ENVS sweep with random
     actions barely scaled — misleading, because it omits the policy.)
  2. **On CPU, the gradient/learn phase dominates (~85%).** Per PPO cycle (4096
     steps): rollout ≈ 2.4 s, learn ≈ 14 s — NatureCNN backprop on CPU, batch 1024.
     → ~250 fps, and CPU *thermally throttles* on long runs (408→255).
  3. **MPS makes learn ~5x cheaper → ~2.5x end-to-end (~629 fps, stable).** So
     `--device auto` now resolves to MPS on Apple Silicon (SB3's own "auto" only
     picks CUDA-or-CPU, so it silently used CPU; resume also ignored `--device` and
     always used CPU — both fixed).
  4. **On MPS the bottleneck flips to the ROLLOUT — specifically per-step policy
     inference** (batch-8 forward pass with CPU↔GPU transfer, 512x/cycle). Confirmed
     because `N_EPOCHS=1` is only ~10% faster than 4 (if learn still dominated it'd
     be ~4x), i.e. learn is now only ~12% of the cycle.
  5. **Therefore the lever on MPS is `N_ENVS`**, which amortizes those transfers
     over a bigger predict batch (same #transfers/cycle, more samples each): 8→629,
     12→779, 16→943, 24→1166, 32→1378 fps. Default bumped 8→**16**. Nearly free for
     learning (for a fixed timestep budget PPO does the same #gradient-updates
     regardless of N_ENVS); cost is rollout-buffer memory + mild sample-efficiency
     risk at very high counts.
  - **Dead ends (tested, don't bother):** `BATCH_SIZE` 512→4096 moved fps ~5%
    (it's a learn-phase knob; learn no longer dominates). `N_EPOCHS` down only ~10%
    and it's a sample-efficiency trade.
  - Re-benchmark anytime: `python -m tools.bench` (env sweep) / `--model <ckpt>`
    (predict split); sweep N_ENVS/batch/epochs via `src.train` flags + `--save-freq
    999999`.
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
7. **Loadout lesson (resolved).** The agent over-invested in Speed, which *hurts* a
   precision navigation task. Reward penalties couldn't stop it (speed is
   net-positive early; the overshoot cost is γ-discounted to ~0) → switched to a
   **hard cap via action masking**. The cap worked (`reward/powerup` −4.25 → +11.29,
   score doubled) but `best_score` stayed at ~280 — so **speed was a real bug but
   not the gauntlet blocker.** Lesson: when a reward penalty can't beat a
   discounted future cost, mask the action instead of pricing it; and a fixed
   behavior that doesn't move the headline metric means you ruled out a cause, not
   solved the problem.
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
- **Resumability gotcha (fixed).** `vecnormalize.pkl` (reward-norm running stats)
  used to be saved *only* at a run's natural end (`model.learn()` finishing) — so a
  Ctrl-C'd or still-running job left mid-run checkpoints with **no** stats, and
  resuming them silently rebuilt fresh `VecNormalize` (policy transfers, but norm
  resets → a transient critic wobble). Fixed: `CheckpointCallback(save_vecnormalize=
  True)` now writes `lifeforce_ppo_vecnormalize_<N>_steps.pkl` beside every
  checkpoint, and `train.vecnorm_for()` resolves it on `--resume` (falling back to
  the canonical `vecnormalize.pkl`). Checkpoints from *before* this fix (e.g. the
  `lv1-speedcap` run) have no stats — resuming them rebuilds fresh, which is
  acceptable for a curriculum drill since that shifts the reward distribution anyway.
- **Phase 2: auto-curriculum.** Designed, not built. A callback that detects
  "stuck" (no new `best_score` in ~N steps), auto-captures the current death point
  to `states/` (envs pick it up live via re-glob), and drills it — automating the
  plateau→capture→drill loop. Build only after manual curriculum demonstrably
  breaks a wall.
- **Does the loadout fix (no speed) break the gauntlet?** **Answered: no.** The
  hard cap fixed the loadout (`reward/powerup` positive, score doubled) but
  `best_score` stayed at ~280. Speed ruled out; gauntlet has a second cause.
- **What IS the gauntlet blocker?** Narrowed. Ruled out: loadout (speedcap_2),
  curriculum/practice (speedcap_3 drilled the wall directly), exploration
  (drill-explore at 0.03 and 0.05, entropy confirmed up). All cheap levers
  exhausted → **structural.** Remaining suspects: (a) **perception** — gap
  unresolvable at 84×84 (`FRAME_SIZE` 128), (b) **control precision** — gap needs
  finer timing than skip-4 allows (`FRAME_SKIP` 2). Disambiguate by watching *how*
  it dies before committing a fresh train.
