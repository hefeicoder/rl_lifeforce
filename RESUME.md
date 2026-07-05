# Session resume state (auto-maintained by Claude — check here after a crash/closed window)

_Last updated: 2026-07-03 ~21:15 (Session 7, branch `go-explore`)_
_Full history: `docs/go-explore-l3-progress.md` (Sessions 1–7)._

## Key facts (stable)

- **Progress metric = steps survived.** `x_pos` (RAM 0x350) is ship SCREEN position
  (15..232), NOT level progress — the game auto-scrolls. No "232 barrier" exists.
- **Full-level bar: 891 steps** (2M baseline `lv1-front-speed/2000000`), still unmet.
- **Eval gotcha:** `tools.segment_search.rollout` overwrites
  `env.unwrapped.initial_state` — save it first and restore before level-start
  evals, else "full level" silently starts from the loaded state.
- Long runs: always `nohup python -u ... > logs/<name>.log 2>&1`.
- Beam searches: deterministic (seed 0) — safe to kill and re-run verbatim.
  NEVER use `--greedy-move` (failed; floods the beam with the baseline's own death).

## Milestones this session

1. **Pinch solved** (go-explore iteration 1): beam demo `demos/l3_pinch_beam198.npz`
   → BC → PPO (`l3-bc-ppo`) → consolidation (`l3-consolidate`, `l3-consolidate2`).
   x120-start survival 107 → 434; full level 75 → 811.
2. **Best full-level policy so far:** `checkpoints/l3-consolidate2/lifeforce_ppo_final.zip`
   (811 steps / score 190). Dies at a step-811 terrain event; watch it live:
   `python -m src.play --model checkpoints/l3-consolidate2/lifeforce_ppo_final.zip --deterministic --episodes 1`
3. **Step-811 escape found** (go-explore iteration 2): beam from
   `states/l3_b232_bd120.state` (step 691) accepted +281 (REPRODUCED 2/2) —
   survives to ~step **1091**. Demo `demos/l3_b811_beam.npz`, handoff
   `states/l3_b811_beam.state`.
4. **BC round 2:** 20 epochs needed (10 too weak): `checkpoints/l3-bc2/lifeforce_ppo_bc20.zip`,
   greedy-from-bd120 = 139 steps (past the death). Full-level temporarily 65 —
   consolidation will restore.

5. **Robustify round 2 (`l3-bc2-ppo`): done.** bd120 up to 532; x120 collapsed
   mid-run, recovered to 434 by 500k. Consolidation base = 500k ckpt.
6. **891 BAR BROKEN (consolidation round 3, `l3-consolidate3`):**
   **NEW BEST = `checkpoints/l3-consolidate3/lifeforce_ppo_400000_steps.zip`**
   — full level **1111 steps / score 699** (baseline 891/190), bd120 374,
   x120 432. Checkpoints 450k+ washed out (~530/90) — ALWAYS sweep.
   Watch: `python -m src.play --model checkpoints/l3-consolidate3/lifeforce_ppo_400000_steps.zip --deterministic --episodes 1`
7. **Iteration 3 started at the step-1111 death.** Frontier states captured
   (`states/l3_b1111_bd{120,80,40}.state`, `l3_b1111_maxx.state`). First beam from
   bd120 (step 991) accepted INSTANTLY: single `UPx4` → +94 (214 total, verified
   2/2) — `demos/l3_b1111_beam.npz` + `states/l3_b1111_beam.state` banked. The
   death is a trivial dodge; terrain after 1111 is easy for a while.

8. **BC on the 4-step demo FAILED** (20 epochs on 4 samples broke the policy:
   bd120 120 → 15; checkpoint deleted). **Lesson: no BC for tiny dodges — PPO
   exploration finds them; BC is for long precise maneuvers (76-80 step demos).**

9. **Deep beam (`l3_b1111_deep`): done.** Global best verified 2/2: 96-step script
   → 218 total (+98). `demos/l3_b1111_deep.npz` + `states/l3_b1111_deep.state`
   (BC-viable fallback; deeper curriculum start).
10. **Drill (`l3-drill1111`, no-BC): WORKED.** Dodge learned (b1111 120→~263-300),
    b811 up to 624, x120 stable ~455, and full-level jumped to **1245-1251** at
    50k/450k/500k even at mix 1.0. Final washed out (504) — sweep, always.
    Best all-around: `checkpoints/l3-drill1111/lifeforce_ppo_500000_steps.zip`
    (full 1245 / b1111 263 / b811 624 / x120 453).

11. **Consolidation round 4: done — NEW FLAGSHIP:**
    **`checkpoints/l3-consolidate4/lifeforce_ppo_400000_steps.zip`** — full level
    **1261 steps / score 708** deterministic from level start (session start:
    891/190). Nearly all checkpoints stable ~1250; skills: b1111 277, b811 564,
    x120 474. Watch:
    `python -m src.play --model checkpoints/l3-consolidate4/lifeforce_ppo_400000_steps.zip --deterministic --episodes 1`

12. **Iteration-4 beam (`l3_b1261_beam`): done.** Global best = round-1 `Rx12`
    → **178 total (+59), cont 166, REPRODUCED 2/2**. `demos/l3_b1261_beam.npz` +
    `states/l3_b1261_beam.state`. 12-step script → drill without BC.

13. **Drill (`l3-drill1261`): done.** Full-level hit **1396 (250k) / 1402 (350k)**
    mid-run; 450k+ overspecialized (x120 → 31, full → ~500). Consolidation base =
    **250k** (b1261 181 / b1111 339 / b811 624 / x120 632 / full 1396).

14. **Consolidation round 5: done — NEW FLAGSHIP:**
    **`checkpoints/l3-consolidate5/lifeforce_ppo_50000_steps.zip`** — full level
    **1398 steps / score 811** (b1261 162 / b1111 315 / b811 707 / x120 583).
    Nearly all checkpoints 1280–1398. Session: 891/190 → 1398/811.
    Watch: `python -m src.play --model checkpoints/l3-consolidate5/lifeforce_ppo_50000_steps.zip --deterministic --episodes 1`

15. **Iteration-5 beam (`l3_b1398_beam`): done.** `DOWNx4 > Rx4 > UPx12` (20 steps)
    → **156 total (+96), REPRODUCED 2/2**. `demos/l3_b1398_beam.npz` +
    `states/l3_b1398_beam.state`. (Reload baseline was 60, not 120 — reloaded
    frame-stack diverges slightly from the level-start trajectory; normal.)

16. **Drill (`l3-drill1398`): done.** Full-level **1508 @450k** (b1398 226 /
    b1261 368 / b811 747 / x120 726 — all strong). 500k+ washed out (~530).
    Consolidation base = 450k.

17. **Consolidation round 6: done — NEW FLAGSHIP:**
    **`checkpoints/l3-consolidate6/lifeforce_ppo_final.zip`** — full level
    **1736 steps / score 1314** (b1398 290 / b1261 261 / b811 796 / x120 732).
    Score jump 811→1314 = an entire new scoring section reached. Session:
    891/190 → 1736/1314.
    Watch: `python -m src.play --model checkpoints/l3-consolidate6/lifeforce_ppo_final.zip --deterministic --episodes 1`

18. **Iteration-6 beam #1 (bd120, coarse): FAILED** — pinned, best +14 < min-gain
    30 after 13 rounds. This hazard resists coarse dodges.
    Log: `logs/l3_b1736_beam.log`.

19. **Iteration-6 beam #2 (`l3_b1736_fine`, bd80 + durations 2/4/8): ACCEPTED
    at round 19** — 68-step precision script → **134 total (+57), REPRODUCED
    2/2**. `demos/l3_b1736_fine.npz` + `states/l3_b1736_fine.state`. The
    step-1736 event is a hazard SEQUENCE needing fine vertical timing — coarse
    4/8/12/16 durations couldn't thread it.
20. **BC round 4** (68-step demo, 20 epochs): partial transfer (77 → 85 greedy;
    demo line 134). Weak-looking BC seeds still carried cycle 2 — proceeding.

21. **Drill (`l3-drill1736`): MISSED the target skill.** b1736 probe never moved
    (~73-89 vs demo 134); retention skills ballooned (x120/b811 hit the 900 probe
    cap) but full-level regressed to 1569. Root cause found by probing the
    handoff state: **fresh reload of a state resets the frame-stack — the policy
    dies in 14-17 steps from the handoff (vs 66 during in-run verification).**
    Handoff states are WEAK curriculum starts (cold stack, precarious position);
    scripted actions are immune (open-loop). Recorded as an eval/curriculum gotcha.

22. **Extended fine beam (`l3_b1736_fine2`, 40 rounds): banked +125** — 178-step
    script, 202 total, REPRODUCED 2/2 → `demos/l3_b1736_fine2.npz`. Still gaining
    ~+10/round at the cap; the gauntlet extends past 200 scripted steps and no
    healthy handoff yet (cont ~24).

23. **Fine beam round 3 (`l3_b1736_fine3`, 60 rounds, cap 400): banked +231** —
    267-step script, 308 total from bd80 (≈ step 1964 from level start),
    REPRODUCED 2/2 → `demos/l3_b1736_fine3.npz`. Was still gaining slowly at
    round 60; continuation ~40 (gauntlet's far side not yet reached).
24. **BC round 5: best transfer of the session** — 20 epochs on the 267-step
    demo → greedy-from-bd80 = **389 steps** (beats the script's own 308).
    `checkpoints/l3-bc5/lifeforce_ppo_bc20.zip`. Long demos clone cleanly.

25. **Drill (`l3-drill1736b`) + consolidation 7: CORRIDOR ERODES.** Drill peaked
    b1736=491 @100k then washed to ~90; consolidate7 peaked full=1629 (< 1736
    bar) with b1736 ~90. Mechanism: the 267-step corridor has near-zero
    tolerance — STOCHASTIC training rollouts die in it constantly, so PPO
    learns avoidance even though the deterministic policy threads it. BC5's
    own retention is wrecked (b811 16, x120 21, full 61): implant vs erosion
    is symmetric. **Flagship still consolidate6/final (1736/1314).**

26. **Low-entropy consolidation (`l3-consolidate8`, ent 0.01): FAILED — mode
    collapse.** All checkpoints play one degenerate 485-step / score-0 line
    from the level start (too little entropy to rebuild the BC-damaged
    skills) and the corridor eroded anyway (389 → 73).

## CYCLE 6 RESCUE IN PROGRESS: demo-ensemble ("mixed") BC

Three variant beams running IN PARALLEL (fine durations 2/4/8, cap 400,
60 rounds, no early accept), for pooling with the banked bd80 line:

- `l3_b1736_v2_bd120` — entry from bd120 (step ~1616), seed 0 → `logs/l3_b1736_v2_bd120.log`
- `l3_b1736_v3_bd40`  — entry from bd40 (step ~1696), seed 0 → `logs/l3_b1736_v3_bd40.log`
- `l3_b1736_v4_seed1` — entry from bd80, seed 1 (tie-break diversity) → `logs/l3_b1736_v4_seed1.log`

ALL FOUR DEMOS BANKED (verified 2/2 each):
- fine3 (bd80, seed 0): +231 → ~step 1964
- v2 (bd120, seed 0): +65 → ~step 1810 (approach coverage)
- v3 (bd40, seed 0): **+440 → ~step 2181, continuation 209 — the gauntlet HAS
  a far side and this line reaches it**
- v4 (bd80, seed 1): +342 → ~step 2075 (different path family)

**Pooled BC (`l3-bc6`, 1032 examples, 20 epochs, loss 0.027): TUBE TEST PASSED**
| seed | bd120 | bd80 | bd40 |
|---|---|---|---|
| no BC | 129 | 77 | 45 |
| BC5 single | 138 | 389 | 26 (WORSE than baseline) |
| **BC6 ensemble** | **194** | **302** | **455** |

Past the death from all three entries — the coverage hypothesis holds at the
BC stage. (Gotcha logged: piping the BC trainer through `head` SIGPIPE-kills
it mid-run; write to a log file instead.)

**`l3-consolidate9` (from BC6, mix 0.3, drill skipped): FAILED —
DIAGNOSTIC.** Tube held at 50k (178/170/189) but eroded by 100k (~100/70/30);
full-level never rebuilt (peak 572, score ~0-15 — the 1032-example BC damaged
general play too much for one consolidation to repair). **Conclusion: erosion
is the PPO objective's doing, not a seed-coverage problem.** Ensemble = better
seed, unchanged erosion dynamics.

**Interleaved self-imitation loop (`tools/si_loop.sh`, NEW TOOL): written,
launched, then STOPPED BY USER at cycle 1** — superseded by the anti-jitter
direction below. The tool remains available (8×[PPO 50k → BC refresh 3ep
lr 1e-4]; cadence matches the measured erosion onset) if needed later.

## NEW DIRECTION (user insight): anti-jitter / prefer stillness

User observation from live play: the ship "vibrates" (random small moves)
~70-80% of the time — deadly in carve-a-path terrain like the step-1736 web
(shoot a channel, then STAY on it). Measured on the flagship (deterministic,
full level): **churn 64%** (movement action changes vs prev step), **HOLD
usage 7%**, near-uniform move distribution = no preference, pure vibration.

Two root causes found:
1. No reward reason to be still — jitter is free; PPO's entropy bonus actively
   prefers spread-out actions where values are flat.
2. **The beam vocab NEVER included HOLD (move 0)** — every demo we ever banked
   is constant-motion, and BC amplifies it. A stay-put solution (if found)
   would also be far more noise-TOLERANT → may dissolve the corridor-erosion
   problem at the source.

Changes landed (committed? not yet):
- `config.py`: REWARD_MOVE_COST (per-step, non-HOLD) + REWARD_CHURN (per-step,
  movement changed vs prev) — defaults 0.0, CLI-driven.
- `env.py`: LifeForceWrapper computes r_move/r_churn (unit-checked exactly);
  new `move`/`churn` entries in reward_components.
- `train.py`: `--reward-move-cost`, `--reward-churn` flags.

**`l3-calm1` (move 0.02 / churn 0.05, ent 0.03): dose too weak.** Churn
63-65% (unchanged), HOLD 7→13-16% mid-run then faded. Harmless though: full
1704-1713 at best ckpts, x120 hit the 900 probe cap repeatedly. Penalties are
safe → escalate the dose.

**`l3-calm2` (move 0.05 / churn 0.2, ent 0.015): WORKING, gradual.** Churn
64.2 → 57.1% falling ~linearly (~1.5 pts/100k), HOLD 7 → 19%, full 1715-1719,
x120 pinned at the 900 probe cap from 250k (BETTER than flagship's 732). Zero
survival cost. Best: `checkpoints/l3-calm2/lifeforce_ppo_500000_steps.zip`.

**`l3-calm3` (churn 0.3, ent 0.01): SUCCESS — NEW FLAGSHIP.**
**`checkpoints/l3-calm3/lifeforce_ppo_500000_steps.zip` — full level 1750
steps / score 1311, churn 47.6% (was 64.2%), HOLD 29.1% (was 7.3%), x120 817.**
First policy to beat the old flagship's 1736 — calmness IS performance.
CAUTION: calm3's `final` checkpoint collapsed into hold-forever (221 steps,
55% HOLD) — churn 0.3 + ent 0.01 is the dose CEILING; do not escalate further.

Dose-response record (full-level trajectory, deterministic):
| stage | churn | HOLD | full |
| flagship | 64.2% | 7.3% | 1736 |
| calm1 (0.02/0.05, ent .03) | ~64% | ~12% | 1704-1713 |
| calm2 (0.05/0.2, ent .015) | 57.1% | 19.1% | 1715 |
| calm3 (0.05/0.3, ent .01) | 47.6% | 29.1% | **1750** |

User eyeball test PASSED ("looks good, died at another pinch point").

## GO-EXPLORE CYCLE 7 (step-1750 death, from the CALM base) — IN PROGRESS

Frontier captured from calm3-500k: `states/l3_b1750_bd{120,80,40}.state`
(steps 1630/1670/1710; the maxx capture is degenerate — x hits 232 at step 28
— IGNORE maxx captures now that x semantics are known).

Cycle-7 events so far:
- bd40 beam: EXHAUSTED round 7 — at step 1710 the corridor is already FORCED
  (any 2-step deviation dies at ~13 vs greedy's 39). Dodge must happen earlier.
- **bd80 beam (HOLD vocab + beam 16): ACCEPTED +107** (161 total ≈ step 1831),
  REPRODUCED 2/2 — **script is ~1/3 pure HOLD** (`UPx4 > HOLDx8 > ... >
  HOLDx8 > ... HOLDx2 > DOWNx4`). The stay-put solution is real; wide beam
  fixed the earlier dilution. Demo `demos/l3_b1750_bd80.npz` (58 steps).
- **GOTCHA (burned): beam `--name` must NOT equal a captured state's name** —
  segment_search saves its handoff to `states/<name>.state` and OVERWROTE the
  bd80 lead-in capture; probes silently measured the handoff. Lead-ins
  re-captured under prefix `l3_c1750_*`. True-lead-in BC probes: base 54,
  BC7-20ep **77** (past death; 5/10-epoch doses worse — dataset size, not
  epochs, is the constraint).

**`l3-calm-cons1` (BC7 seed + penalties DURING formation): EROSION BEATEN.**
The c1750 corridor probe held 72-130 for the ENTIRE 500k run (base 54, seed
77; cycle-6 corridors collapsed 490→90 within 300k). b1736 probe held too.
Churn fell further to ~35%, HOLD ~41%. Remaining: full-level only rebuilt to
1445/712 (bar 1750) with an odd low-score path variant at some ckpts; x120
wobbly. Trend strongly upward → keep consolidating.

**`l3-calm-cons2`: CYCLE 7 COMPLETE — NEW FLAGSHIP.**
**`checkpoints/l3-calm-cons2/lifeforce_ppo_final.zip` — full level 1753 /
score 1212, churn 41%, HOLD 37%, corridor probe 109, x120 at the 900 cap.**
The 500k ckpt holds the raw survival record: **1805 steps** (first past 1800;
session start was 891 — >2x). Corridor probes held through BOTH 500k
consolidations — the no-drill calm recipe does not erode. Note: score 1212 vs
the old 1311 (slightly different path); steps are the progress metric.

## CYCLE 8 (branch `cycle8-boss-approach`, merged main at d4daf7a) — RUNNING

Boss check at the 1753 death: scroll clock STILL TICKING (RAM 0x2F advancing
through death; stage_num 0) → still terrain, boss further out. Score frozen
at 1212 since ~step 1500 = hazard-dense, enemy-light precision stretch.

Frontier captured: `states/l3_d1753_bd{120,80,40}.state` (from
calm-cons2/final's 1753-step run).

CYCLE 8 COMPLETE:
- Beam: +93 verified (180 total ≈ step 1853), 66-step HOLD-rich script →
  `demos/l3_d1753_beam.npz`.
- BC8: 87 → 98 (past death). Consolidate ×2 (calm recipe): corridors held
  ALL RUN both rounds (4th consecutive erosion-free consolidation).
- **NEW FLAGSHIP: `checkpoints/l3-c8-cons2/lifeforce_ppo_final.zip` —
  1762 steps / score 1218, churn 37.6%, all probes healthy** (d1753 92,
  c1750 108, x120 900-cap). c8-cons2/50k briefly hit **1810** (absolute
  record) but with x120 broken — footnote only.
- Caveats: smallest cycle gain yet (+9 flagship-to-flagship); full-level
  plateaus ~1770 while the demo line reaches ~1853 — a fresh hazard sits
  right behind the banked corridor. x120 probe is BIMODAL across checkpoints
  (35↔900) — the pinch behavior toggles between two modes; worth a dedicated
  look if it persists.

CYCLE 9 COMPLETE: beam banked a 12-step dodge (`DOWN+Rx8 > HOLDx4`, +54,
verified) → NO BC (tiny-dodge rule) → consolidate ×2 (calm recipe).
**NEW FLAGSHIP: `checkpoints/l3-c9-cons2/lifeforce_ppo_400000_steps.zip` —
1836 steps / 1215, all probes healthy (e1762 128, d1753 132, x120 900),
churn 35.5%.** Record: **1856** (c9-cons2/500k, x120 weak). +74
flagship-to-flagship — conversion recovered after cycle 8's +9.

## CYCLE 10 (death ~1836): FLAT — flagship unchanged
Boss check: scroll still ticking at 1836 (terrain). Beam banked only a
14-step micro-dodge (+32, `demos/l3_f1836_beam.npz`); two calm
consolidations peaked at 1830-1831 — **did not beat c9's 1836.** The
1820-1856 band has now absorbed ~1M steps of consolidation without moving.

**FLAGSHIP remains `checkpoints/l3-c9-cons2/lifeforce_ppo_400000_steps.zip`
(1836/1215).** Record remains 1856 (c9-cons2/500k).

## CYCLE 11 (LOADOUT — user insight): items make late channels easier

User observations from live play: (1) HOLD improvement visible and good;
(2) Missile/Options would widen the carved channels late-level; (3) most
capsules are at the BEGINNING of the level; (4) drop HOLD from farm-search
vocab. Measurements confirmed everything: the calm flagship NEVER farms
(0 items all run; 733 wasted A-presses), while pre-calm checkpoints
(2M-baseline, consolidate6) bought missile+speed4 by step 600 — **the calm
training washed the farming habit out.**

Tooling added to segment_search (4 measured design iterations):
- ACT pseudo-move (HOLD + press A for 1 step; +1 vocab entry, not 2x)
- --rank-powerups: loadout-first ranking. pu = powerbar + 5*missile +
  8*options + 7*shield. Speed weighs 0 (v2: beam farmed the cheapest slot);
  powerbar +1 is the progress breadcrumb (v3: unspent capsules were invisible
  and the search had to find eat->eat->ACT blind); purchases strictly beat
  hoarding (missile 5>2, option 8>5, shield 7>6).
- Loadout measured AT HANDOFF, not episode end (v4: the continuation's
  A-mashing spent the script's bank on speed, erasing the ranking signal).
- GREEDY segments now SUPPRESS the policy's own A-press (v6: ride the old
  policy's chase/kill/collect expertise, keep all spending in ACT segments).

**Farm demo BANKED (`demos/l3_farm_v6.npz`, REPRODUCED 2/2):** ~430-step
opening from level start on the consolidate6 base — 2 missiles bought via
deliberate cursor-timed ACTs + 2 capsules banked at handoff, 552 total steps.
`states/l3_farm_v6.state` = first curriculum state WITH a loadout.

**RUNNING: `l3-c11-cons1`** — BC9 (farm demo into flagship; opening broken to
59 as usual pre-consolidation) → consolidation at **HALF penalty dose**
(churn 0.1, move 0.02) because chase-farming is churn-heavy and full dose is
the suspected habit-killer. Curriculum: {farm_v6 handoff, f1836/e1762/d1753
bd80s, x120}. Sweep MUST include: loadout@600 on full runs, corridor probes
(watch for erosion regression at half dose), churn/HOLD.

## (older) PAUSED FOR DIRECTION — options for the ~1850 plateau:
(a) Deeper script: beam from bd80 with --script-cap 600+ and more rounds —
    maybe the zone needs a long carried line like the 1736 web (v3-style).
(b) Ensemble: 2-3 demo variants (bd120/seed-1) pooled BC — worked for
    coverage before; cheap now.
(c) Investigate first: trace WHERE the 1830 runs die vs what the demos dodge
    (are we solving the wrong hazard? is trajectory divergence eating the
    dodge?), and check probe oscillation (44↔160 on e1762 across ckpts).
(d) Longer consolidations (1M steps/round) — the band may just need more
    rebuild time per skill.
2. **`l3_b1736_hold` beam: DONE — negative result.** Accepted only +53
   (38-step script, REPRODUCED 2/2, `demos/l3_b1736_hold.npz`) and the final
   line contains NO HOLD segments; beam exhausted at round 30 (vs v3's +440
   at 60). Mechanism: 6th move inflated branching 20%/segment at the same
   beam width → shallower coverage. **Lesson: expanding the vocab needs a
   wider beam (e.g. --beam 16) to be a fair test.** The stay-put hypothesis
   now rides on the reward-side test (calm1).

## CYCLE 6 STATUS: BLOCKED ON ROBUSTIFICATION (superseded by the rescue above)

**Flagship: `checkpoints/l3-consolidate6/lifeforce_ppo_final.zip` (1736/1314).**

What's banked and reusable: verified 267-step gauntlet script
(`demos/l3_b1736_fine3.npz`, +231, ≈ step 1964 from level start) and the BC5
seed that threads it greedily for 389 steps
(`checkpoints/l3-bc5/lifeforce_ppo_bc20.zip`).

Why it's stuck: the corridor has near-zero tolerance. ANY stochastic PPO
training dies in it constantly → learns avoidance (drill: 491→90;
consolidate7: ~90). BC implants it but wrecks other skills (b811 16, x120 21).
Low entropy prevents rebuilding (mode collapse). Erosion vs implant is
currently a strict trade-off.

Ideas for next session (in rough order of promise):
1. **Widen the corridor tolerance**: beam-search demo VARIANTS from perturbed
   starts (bd120/bd80/bd40, different warmups) → BC the ensemble (DAgger-ish
   coverage) so nearby states also have good actions.
2. **KL-anchored consolidation**: consolidate from BC5 with a KL penalty to
   the BC5 policy (or interleave 1-2 BC refresh epochs every ~50k PPO steps —
   true self-imitation) so the corridor can't drift far.
3. **Reward shaping**: small dense bonus for scroll-clock progress past the
   step-1736 mark (or death penalty scaled by depth) so corridor attempts
   aren't pure negative signal.
4. Entropy schedule: start 0.03 (rebuild), anneal to 0.005 (protect corridor).

## Next steps

1. (user decision) pick a rescue idea above, or accept 1736 and move on.
2. Update PR #1 / progress doc with the cycle-6 saga; commit.

## Next steps (in order)

1. When the beam banks: drill (mix 1.0, curriculum = {b1736_bd120, new handoff,
   x120, b232_bd120, b1111_bd120, b1261_bd120, b1398_bd120}; BC only if script
   ≥30 steps) → consolidate (mix 0.3) → sweep vs bar **≥1736**.
2. Repeat at the next death. Cycle ≈ 30–40 min.
3. Commit the session's work (progress doc, segment_search.py, RESUME.md, demos/
   states) — propose to user first.
