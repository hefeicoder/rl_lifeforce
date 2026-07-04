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

## CYCLE 6 STATUS: BLOCKED ON ROBUSTIFICATION (auto-loop paused here)

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
