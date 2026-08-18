# Project handoff and experiment log

_Last updated: 2026-08-16 (branch `cycle8-boss-approach`)_
_Full history: `docs/go-explore-l3-progress.md` (Sessions 1–7)._

## Key facts (stable)

- **Progress metric = steps survived.** `x_pos` (RAM 0x350) is ship SCREEN position
  (15..232), NOT level progress — the game auto-scrolls. No "232 barrier" exists.
- **Original full-level bar: 891 steps** (2M baseline
  `lv1-front-speed/2000000`). Current flagship **clears Level 1 in 3642
  single-life steps**.
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

CYCLE 11 COMPLETE:
- cons1 (half-dose penalties): farming transferred — policies bank 6 capsules
  unprompted; 450k hit 1860/1408 (records) but nothing was ever SPENT (2 ACT
  frames among 430 were too weak a BC signal for purchase timing).
- **AUTO_BUY added to the env** (config flag, on): Discretizer presses A by
  rule at MISSILE_SLOT(2, if none) / OPTION_SLOT(5, if <2) and IGNORES the
  agent's activate head (vestigial, like hardwired B). Measured necessity:
  agent A-mashing spent the bank at random slots (Laser@4).
- cons2 (armed training): **every checkpoint fields Missile+Option by step
  1200**; armed-vs-unarmed gap closed (1743→1846).
- **NEW FLAGSHIP: `checkpoints/l3-c11-cons2/lifeforce_ppo_350000_steps.zip`
  — 1846 steps / score 1344 (score record 1444 @final), armed, corridors
  held.** Half-dose penalties (churn 0.1/move 0.02) are the new standard while
  farming coexists with calm.

## THE 1UP BUG (found via farming; env FIXED; records restated)
Farming pushed scores past the 1UP threshold (~score 1200, ~step 1400);
the death check (`lives < start_lives`) then missed the next death and
episodes continued on an UNARMED RESPAWN (~380 ghost steps). Fixed in
env.py: death = any lives DECREASE. TRUE single-life records: c8 1446,
c9 1444, **c11-armed 1462 (best)**. The "1820-1860 plateau" was mostly the
respawn tail, and cycles 10-12's first captures (f1836/g1846) were
post-death ghost states — that's why those beams only found micro-dodges.
GOTCHA: captures made before the fix are suspect; only `l3_h1462_*` and
older (<step ~1400) captures are clean.

## CYCLE 12 — TRUE armed frontier attack: BEAM BANKED, PAUSED FOR USER
Clean captures from the armed flagship's true death (1462):
`states/l3_h1462_bd{120,80,40}` (bd80 = step 1382, verified armed:
missile+option in-state, pu=13).

**BEAM RESULT (the project's largest): +405 verified 2/2** —
`demos/l3_h1462_beam.npz` (~230-step script, HOLD-sprinkled) +
`states/l3_h1462_beam.state`. 485 total from bd80 ≈ **step 1867 single-life
from level start**, continuation ~280 (healthy far side). The armed policy
sails through terrain that capped every unarmed search at +107. USER
HYPOTHESIS (options widen the channels): CONFIRMED decisively.

CYCLE 12 COMPLETE (the biggest cycle of the project):
- BC10: 80 → 288 greedy from the true lead-in (2nd-best transfer ever).
- Consolidate ×2 (armed, half-dose): corridor NON-ERODING (h1462 probe
  232-401 throughout both runs), full-level rebuilt then surpassed.
- **NEW TRUE FLAGSHIP: `checkpoints/l3-c12-cons2/lifeforce_ppo_final.zip` —
  1778 single-life steps / score 1374 / churn 47.8% / mis+opt fielded /
  corridor 401 / all retention probes healthy.** +316 vs the honest 1462.
- Boss check at 1778: scroll still ticking, stage 0 — terrain. The +405 demo
  line reached ~1867, so the next hazard sits between 1778 and ~1900.

CYCLE 13 COMPLETE: beam banked an 18-step dodge (+83, no BC) → consolidate
×2 (armed, half-dose). **NEW FLAGSHIP: `checkpoints/l3-c13-cons2/lifeforce_ppo_final.zip`
— 1836 single-life / score 1362** (records: 1854 full @c13-cons1/300k,
score 1483 @c13-cons1/250k). All probes healthy (i1778 138, h1462 419,
x120 784), mis+opt fielded. Boss check at 1836: scroll still ticking —
terrain. Honest trajectory: 1462 → 1778 → 1836.

CYCLE 14 COMPLETE: beam +213 verified (~150-step HOLD-dense script, reaches
~step 2069); BC11 99→181; consolidate ×2.
**FLAGSHIP: `checkpoints/l3-c14-cons2/lifeforce_ppo_50000_steps.zip` —
1833 single-life / 1368, corridor j1836=176, all probes healthy.**
Record: 1856 @c14-cons2/350k (corridor weaker there). Boss check at 1856:
scroll still ticking — terrain. Honest march: 1462→1778→1836→1856.
NOTE: the new corridor erodes across cons rounds (181→176→84 by final) —
the j-corridor is tighter than its predecessors; if cycle 15's sweep shows
the same, consider full-dose penalties for one round or an ensemble demo.

CYCLE 15 (the si-loop experiment):
- Beam banked the PROJECT RECORD: **+423 verified** (448 total from bd80 ≈
  step 2201 from level start — first line past 2000, inside the boss window).
  `demos/l3_k1833_beam.npz`.
- BC12: best transfer ever (25 → 397 of the demo's 448).
- Plain consolidation ×2 eroded the corridor again (397→100→65; full 1818).
- **si_loop (first real use): one balanced winner, then oscillation.**
  Cycle-1 refresh = the keeper. Later cycles pinned the corridor at ~399 but
  collapsed full-level (~600); extension cycles oscillated (1418 → 501)
  rather than converging. Measured drift-shrink per refresh (0.47 → 0.0002)
  shows the corridor DOES stabilize — but 50k PPO chunks then can't rebuild
  general play before the next refresh re-biases. Lesson: **use si_loop with
  1-2 cycles as a post-consolidation corridor patch, not as a long loop.**
- **FLAGSHIP: `checkpoints/l3-si15-1/lifeforce_ppo_bcref.zip`** — full 1827 /
  score 1377 / churn 34% / mis+opt 1/1 / k-corridor 215 (vs 65 before) /
  h1462 412 / x120 651. Ties c14's full with the new corridor 3x stronger.

CYCLE 16 COMPLETE: beam +328 verified (~step 2155 line); BC13 80→279;
consolidate ×2. **FLAGSHIP: `checkpoints/l3-c16-cons1/lifeforce_ppo_300000_steps.zip`
— 1859 single-life / score 1484 (both records among balanced ckpts), x120
900, all healthy.** Raw record: **1908** @c16-cons2/300k (but round 2
collapsed the pinch probe to ~25 across almost every ckpt — regression, not
used). Boss check at 1908: scroll still ticking. Honest march:
1462→1778→1836→1859 (record run 1908).

CYCLE 17 COMPLETE: beam +297 (~step 2155 line); BC14 79→240; consolidate
×2. **FLAGSHIP: `checkpoints/l3-c17-cons2/lifeforce_ppo_500000_steps.zip` —
1866 single-life / score 1493 (both records)**; boss check at 1866: still
terrain. si-patch (2 cycles) tried on it: pinned the n-corridor (375) but
seesawed h-corridor/full — no balanced winner; flagship unchanged.
**KEY DIAGNOSIS: probe-vs-trajectory divergence.** Corridor probes replay
near demo-perfect (~375) while full runs die ~1860 — the true trajectory
reaches the corridor region in a different state than the bd80 capture, so
the skill doesn't engage. Reload-based drills inherently chase a moving
trajectory; the loop's per-cycle re-capture is the working mitigation.
Honest march: 1462→1778→1836→1859→1866.

CYCLE 18: FLAT. Beam +304 (~step 2177 line, verified); BC15 87→218;
consolidate ×2 never beat 1866 (best 1842 @cons1/250k WITH corridor 367;
round 2 collapsed the corridor to 14). Flagship unchanged.
Two flat cycles → variable changed for cycle 19: base on the
CORRIDOR-HOLDING checkpoint (c18-cons1/250k, 1842/corridor 367) instead of
the highest-full one; fresh captures (`l3_q1842_*`) from ITS trajectory.

CYCLE 19: beam +352 verified; BC16 56→272; consolidate ×2 FAILED the 1950
bar (best 1879 w/ broken pinch; balanced ~1830-1847). Per agreement →
**PIVOT: GOLDEN-RUN METHOD — immediate success.**

## GOLDEN-RUN METHOD (the new standard for converting demos)
Build ONE continuous trajectory: base policy plays from level start to the
capture step, then the verified beam script's ACTIONS execute, then greedy
to death — recording (obs, action) every step. Zero capture-state mismatch
(warm frame stack, on-trajectory obs). BC the whole thing back into the
policy that generated it (10 epochs).
- Construction note: the script died 22 steps earlier in the continuous run
  than from the reload (2000 vs 2170 equiv) — directly measuring the
  reload mismatch (likely small alignment offset; try ±1-2 step offsets
  next time to recover the tail).
- Result: **PERFECT CLOSED-LOOP CLONE — first BC pass, no consolidation:
  `checkpoints/l3-golden/lifeforce_ppo_bc10.zip` = 2000 single-life steps /
  score 1396 / loadout 1/1 / h1462 probe 399 / x120 277 / churn 25%.**
  +134 over the old record in one step. FIRST >2000 CHECKPOINT.
- Boss check at 2000: scroll still ticking (242), stage 0 — terrain.

## THE RELOAD-WORLD DISCOVERY (cycle 20 — changes everything)
Cycle 20's beam (+102 from the r2000 reload) did NOT transfer: the same
script actions from the perfectly-aligned continuous step (scroll-clock
verified) die at the old step 2000. Even a reload of a level-start snapshot
gives a different trajectory (dies 805 vs 2000). **Conclusion: em.set_state
reloads land in a frame-phase-shifted world; scripts found there exploit
enemy timings the real run never sees. ALL reload-based search gains past
~step 1800 were unconvertible for this reason.**
Fix shipped in segment_search: `--state RESET` = canonical env.reset()
world + `--warmup N` (greedy actions from level start, replayed from a fresh
reset per rollout) = search happens ON the true trajectory; gains convert 1:1
via golden-build.
GOTCHA: --script-cap counts warmup — set it to warmup+desired script length.

## CYCLE 21: THE 2000-STEP CEILING + GOLDEN2 — COMPLETE
The continuous-world beam exhausted at round 21 with every viable line tied at
exactly 2000. Root cause: `MAX_EPISODE_STEPS=2000` truncated the baseline, so a
search requiring baseline +30 was mathematically unable to accept. The ceiling
is now 5000, and `segment_search` aborts if its baseline hits that ceiling.

With the ceiling lifted, `l3-golden/bc10` actually survives **2152** steps before
a real life loss. At ~2100 it pins itself at x=232 and vibrates UP+R/DOWN+R until
terrain reaches it. A canonical-reset probe found the whole correction:
**UP+LEFT x8 at step 2090 → 3214 steps (+1062), reproduced 2/2**.

Golden recording support now optionally includes the greedy continuation. The
banked full continuous trajectory is `demos/l3_golden_3214.npz` (3214 examples).
BC 10 epochs converged 0.084→0.001 and produced the new flagship:
**`checkpoints/l3-golden2/lifeforce_ppo_bc10.zip` — 3214 single-life steps /
score 1932 / Missile+Option / churn 19.3% / HOLD 68.6%.** It reproduces the
golden run exactly from a cold canonical reset. Retention improved: h1462
399→412 and x120 277→900-cap; p1866 stayed 37 (a reload-world probe).

Visual check corrected the RAM-only inference: step 3214 is already inside the
rotating-brain boss fight. The scroll clock continues incrementing in this arena,
so "clock stopped = boss" is not a valid detector.

## CYCLE 22: LEVEL 1 CLEARED
Canonical boss search + low-rate golden cloning walked the policy around the
rotating arms. The important conversions were 3214→3295→3410→3456→3487→3530,
using full-trajectory demos and **5 BC epochs at lr=1e-4**. The standard 3e-4
dose regressed; the gentler dose reproduced each winning line exactly.

Final probe from the 3530 policy: `UP+LEFT x8` at step 3508 kills the boss and
transitions at **step 3642**, verified 2/2 as a scripted canonical run. The full
clear demo is `demos/l3_boss3530_single.npz` (3642 examples).

**LEVEL-1-CLEAR FLAGSHIP:**
`checkpoints/l3-level1-clear/lifeforce_ppo_bc5_lr1e4.zip`
- canonical deterministic clear: **3642 steps / score 2932 / no life loss**
- verified **3/3** cold resets, identical result
- Missile+Option fielded; churn 19.0%; HOLD 62.9%
- clear signal confirmed: `stage_vertical` **0→1** while `stage_num` stays 0
- proof video: `videos/l3_level1_clear.mp4`
- public release: [Level 1 Clear v1](https://github.com/hefeicoder/rl_lifeforce/releases/tag/level1-clear-v1)

Tooling added during the clear: 5000-step safety ceiling; search abort on a
truncated baseline; full-continuation golden recording; deterministic cached
warmup replay; fast prefix replay that rebuilds the final four-frame stack; and
parallel beam candidate scoring (`--workers`).

The checkpoint and proof video are preserved in the `level1-clear-v1` GitHub
release. Before any robustness PPO, evaluate stochastic clear rate separately;
do not overwrite the deterministic release checkpoint with a later checkpoint.
Level 2 is now a state/config task.

## LEVEL 2 BOOTSTRAP: FIRST SPECIALIST

Captured the real Level-1→2 transition with `tools.capture_stage_start`. The
immediate transition snapshot was too early (`control=1`, lives still settling),
so the tool now holds neutral input through the fly-in and captures after two
stable `control=3` steps.

Canonical Level-2 reset: `states/l2_start.state`

- transition at Level-1 step 3642; settled capture at step 3663
- start RAM: stage `0`, vertical `1`, control `3`, lives `4`, x/y `64/142`
- loadout: Missile + one Option, speed 0
- raw emulator-state SHA-256:
  `41dca197a8c18a0e627c62a6f26f9707ebf8b1505a03ec285a9dd378b7730146`

Frozen Level-1 model from this reset: **233 steps / score 2974 (+42)**, identical 3/3,
churn 14.2%, HOLD 80.3%. Video showed competent opening play and one repeatable
enemy/terrain-cluster death, so PPO was not indicated.

Canonical search (`warmup=173`) accepted `DOWN+LEFT x16` in beam round 1:
**362 steps (+129), continuation 173, reproduced 2/2**. Full golden demo:
`demos/l2_s233_canonical1.npz` (362 examples).

**LEVEL-2 FLAGSHIP:**
`checkpoints/l2-golden-362/lifeforce_ppo_bc5_lr1e4.zip`

- full-golden BC: 5 epochs, lr `1e-4`, loss 0.5755→0.0188
- cold Level-2 reset: **362 steps / score 3007 (+75)**, identical 3/3
- churn 20.5%, HOLD 71.8%
- proof video: `videos/l2_golden362.mp4`
- checkpoint SHA-256:
  `0eece1eac3b036bab6be93ebeb80580fe8ef0d842326f006641eed3925c19d88`

**Bootstrap decision:** keep PPO deferred. Continue canonical search + full-golden BC from
the step-362 frontier. Preserve the Level-1 release and this Level-2 checkpoint.

## LEVEL 2 CANONICAL ADVANCE: 362 -> 1778

On 2026-08-17, six consecutive canonical search + full-golden BC conversions
advanced the Level-2 specialist without PPO:

| Input | Warmup | Verified correction | Search result | BC cold reset |
|---:|---:|---|---:|---:|
| 362 | 282 | `LEFT x16 > LEFT x16` | 505 (+143), cont 191 | 505, 3/3 |
| 505 | 425 | `LEFT x4` | 637 (+132), cont 208 | 637, 3/3 |
| 637 | 557 | `DOWN+LEFT x8 > UP+LEFT x16 > LEFT x4` | 1059 (+422), cont 474 | 1059, 3/3 |
| 1059 | 979 | `UP x16` | 1368 (+309), cont 373 | 1368, 3/3 |
| 1368 | 1288 | `DOWN+RIGHT x2` | 1560 (+192), cont 270 | 1560, 3/3 |
| 1560 | 1480 | `UP+RIGHT x8 > DOWN+LEFT x8` | 1778 (+218), cont 282 | 1778, 3/3 |

All search winners reproduced 2/2 before recording. Every BC conversion used the
full reset-to-continuation trajectory, 5 epochs, and lr `1e-4`; losses converged
respectively to 0.0174, 0.0048, 0.0054, 0.0030, 0.0010, and 0.0007.

**CURRENT LEVEL-2 FLAGSHIP:**
`checkpoints/l2-golden-1778/lifeforce_ppo_bc5_lr1e4.zip`

- cold Level-2 reset: **1778 steps / score 3148 (+216)**, identical 3/3
- churn 11.1%, HOLD 81.4%
- proof video: `videos/l2_golden1778.mp4`
- final golden demo: `demos/l2_s1560_canonical7.npz` (1778 examples)
- checkpoint SHA-256:
  `6b7df9accb064467ac9862270558ba467d1409e047f1fef5ad91942e3eeed36a`

The session improved 362 -> 1778 (4.9x) and the inherited Level-1 baseline
233 -> 1778 (7.6x). PPO remains deferred because every observed failure was
localized, search found a short correction, and the cloned policy retained long
autonomous continuations exactly.

## Next steps

Follow the dedicated [`docs/level2_training_playbook.md`](docs/level2_training_playbook.md).

1. Inspect the step-1778 death and run canonical search from roughly 60-120 steps
   before it using `--initial-state states/l2_start.state --state RESET`.
2. Confirm whether the scroll clock remains a useful Level-2 progress diagnostic.
3. Confirm Level 2's clear signal when reached; never infer it from screen position.
4. After the standalone specialist clears, test the true continuous Level-1→2
   frame-stack/model-switch handoff.
