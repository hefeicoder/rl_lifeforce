# Level 2 vertical-stage training playbook

This is the working plan for extending the released Level-1 policy into Level 2.
It condenses the Level-1 experiment history into the smallest workflow that
actually worked, while accounting for Level 2 scrolling vertically from bottom
to top.

Use [`RESUME.md`](../RESUME.md) for the full experiment chronology and
[`go-explore-l3-progress.md`](go-explore-l3-progress.md) for historical failure
analysis. This document is the operational reference for new Level-2 work.

## Guiding rule

Use **PPO for broad competence**, **canonical search for precise discovery**, and
**full-run behavior cloning for incorporating discoveries**.

The preferred loop is:

1. cold baseline from the real Level-2 reset;
2. diagnose the shape of the failure;
3. canonical `RESET + warmup` search for a localized hazard;
4. record the complete reset-to-continuation golden trajectory;
5. behavior-clone conservatively;
6. accept only repeated cold-reset improvement.

PPO drilling, consolidation, demo ensembles, and self-imitation refreshes are
conditional recovery tools, not automatic stages.

## Confirmed facts versus Level-2 assumptions

### Confirmed

- The Level-1 release checkpoint clears in 3642 agent steps, 3/3 cold resets.
- Level 2 begins with RAM `stage_vertical` (`0x0040`) changing from `0` to `1`.
- `x_pos` (`0x0350`) and `y_pos` (`0x032F`) are screen coordinates, not world
  progress.
- The auto-scroll clock (`0x002F`) was input-independent on Level 1, but wraps at
  256.
- Save-state reloads can change emulator/frame phase and reset the observation
  frame stack. Late Level-1 scripts found in that world did not transfer to the
  real reset trajectory.
- The final working conversion method was canonical search followed by full-run
  golden recording and low-rate BC.

### Must be measured before training

- The exact Level-2 integration/reset state and its initial RAM signature.
- Whether `0x002F` remains a useful scroll clock in the vertical stage.
- The RAM change that corresponds to a real Level-2 clear. A change from vertical
  back to horizontal is plausible, but must not be assumed without video and RAM
  confirmation.
- Whether the released policy's Missile + Option auto-buy rule remains the best
  Level-2 loadout.
- Whether 5000 agent steps is a sufficient safety ceiling for Level 2.
- The policy's natural vertical positioning, churn, HOLD rate, and first true
  single-life death.

## What changes in a vertical stage

Level 1 scrolled horizontally, so several experiments used the ship's screen-x
position as a proxy for tactical forward intent. That logic does not transfer
directly to Level 2.

| Concern | Level-2 rule |
|---|---|
| Progress | Prefer stage clear, single-life survival steps, and a confirmed scroll-clock delta. Do not use screen position as world progress. |
| Position shaping | Start with `REWARD_XPOS=0` and `REWARD_XMAX=0`; they encode an old horizontal retreat problem. |
| Vertical shaping | Do not automatically replace x shaping with y shaping. Staying near the top may help in one section and be fatal in another. Add it only after diagnosing a repeatable retreat/camping basin. |
| Search actions | Keep all nine movement actions, including diagonals and HOLD. Let search discover whether UP, lateral movement, or waiting is correct. |
| Search timing | Continue using fine durations `2 4 8`, adding `16` for sustained movement. Vertical scrolling does not remove timing hazards. |
| Calm behavior | Retain HOLD and monitor churn. A vertical stage may require more lateral dodging, so penalize needless changes rather than assuming movement itself is bad. |
| Loadout | Keep Missile + Option for the first baseline, then verify visually and through survival comparisons. |

For vertical diagnostics, record `min_y`, `max_y`, `mean_y`, and terminal `y` in
addition to the existing step/score metrics. These are diagnostics, not reward
targets.

## Phase 0: establish the canonical Level-2 world

Do this before PPO, curriculum capture, or search:

1. Preserve the Level-1 release checkpoint unchanged.
2. Add or select a real Level-2 stable-retro integration state and make it the
   environment's default reset state.
3. Confirm reset begins with `stage_vertical=1`, active player control, expected
   lives, and a coherent power-up state.
4. Record a short no-op/deterministic video and RAM trace.
5. Verify that repeated resets produce the same initial trajectory.
6. Run until a genuine death or clear and distinguish `terminated`, `truncated`,
   and life loss.

An ad-hoc `.state` loaded with `--from-state` is useful for diagnosis, but it is
not the final canonical world. `segment_search --state RESET` must reset into the
actual Level-2 integration before its results are trusted.

## Phase 1: baseline before changing training

Evaluate the released checkpoint deterministically from Level 2 at least three
times. Capture:

- single-life steps and score;
- termination reason and whether the time limit was reached;
- scroll-clock trace and stage RAM;
- x/y position distributions;
- churn and HOLD percentage;
- Missile, Options, speed, and capsule spending;
- video of the first repeatable death.

Do not change rewards merely because performance is initially weak. First classify
the failure:

| Observed failure | First tool |
|---|---|
| Generally poor shooting/navigation or an unfamiliar mechanic | Short PPO adaptation from the real Level-2 reset |
| One repeatable localized death while general play is good | Canonical beam search |
| A tiny precise correction is known | Canonical search + full golden recording; the full dataset makes local-demo length irrelevant |
| Excessive movement changes or vibrating in place | Moderate calm PPO penalties |
| Capsules collected but loadout is poor | Adjust the deterministic auto-buy rule before asking PPO to learn sparse purchase timing |
| Search winner works only from a loaded state | Reject it; repeat in `RESET + warmup` |
| Deterministic policy is good but stochastic runs fail | Preserve the deterministic checkpoint, then optionally run robustness PPO |

## Phase 2A: PPO only for broad adaptation

Use PPO when the problem is distributed across the level rather than one narrow
action window.

Level-2 PPO should initially:

- start only from the real Level-2 reset;
- keep reward normalization, LR annealing, and `target_kl=0.02`;
- keep death as any life decrease and terminate on it;
- keep survival and score as the primary signals;
- disable old horizontal x-position shaping;
- use short measured runs and sweep intermediate checkpoints;
- avoid overwriting either the Level-1 release or the current best Level-2
  deterministic checkpoint.

If churn returns, begin with approximately:

```text
reward_move_cost = 0.02
reward_churn     = 0.10
ent_coef         = 0.015
```

Escalate toward `0.05 / 0.20` only if measured churn remains harmful. Do not
repeat the `churn=0.3` plus low-entropy recipe that eventually collapsed into
holding forever.

Save-state curriculum is acceptable for a broad, noise-tolerant recovery skill.
It is not the default for late precision corridors, and every claimed gain still
requires a cold Level-2 reset evaluation.

## Phase 2B: canonical search for a localized hazard

Start the search roughly 60-120 agent steps before the repeatable failure. The
warmup remains part of the real reset trajectory:

```bash
python -m tools.segment_search \
  --model checkpoints/<level2-best>.zip \
  --state RESET --warmup <steps-before-hazard> \
  --name l2_golden_<frontier> \
  --beam 8 --rounds 20 --moves 0 1 2 3 4 5 6 7 8 \
  --durations 2 4 8 16 --continuation 800 \
  --script-cap <warmup-plus-script-budget> \
  --record-continuation --workers 4
```

Important gates:

- The greedy baseline must not be truncated.
- Search improvement must reproduce 2/2 from fresh canonical resets.
- `--script-cap` includes the warmup length.
- HOLD remains in the vocabulary. Increase beam width if added branches dilute
  useful HOLD lines.
- Do not use `--greedy-move`; it previously flooded the beam with the baseline's
  own failure behavior.
- Search success means a better continuous trajectory, not a save-state-only
  probe result.

## Phase 3: full golden recording and conservative BC

The dataset must contain the whole trajectory:

```text
Level-2 reset -> policy warmup -> corrective script -> greedy continuation
```

This is why the old rule "do not BC a tiny demo" no longer needs a separate
branch: an 8-step correction embedded in a multi-thousand-step golden run still
contains thousands of retention examples.

Start conservatively:

```bash
python -m tools.self_imitation \
  --model checkpoints/<level2-best>.zip \
  --demos demos/l2_golden_<frontier>.npz \
  --out checkpoints/l2-next/lifeforce_ppo_bc.zip \
  --epochs 5 --lr 1e-4
```

If the maneuver does not become greedy, increase epochs from 5 to 10 before
raising the learning rate. The Level-1 boss sequence showed that `3e-4` could
distort a good trajectory while `1e-4` cloned it exactly.

Do not automatically run PPO after BC. First evaluate the cloned checkpoint from
the real reset. Full golden BC repeatedly advanced the final Level-1 policy with
no consolidation step.

## Phase 4: acceptance and artifact policy

A Level-2 checkpoint becomes the new working best only if it:

1. improves the full cold-reset trajectory;
2. preserves single-life behavior and required loadout;
3. does not terminate by the step ceiling;
4. reproduces deterministically across repeated resets;
5. has a proof video for major milestones.

For a Level-2 clear, require at least 3/3 deterministic cold resets and confirm
the transition visually and in RAM before declaring success.

Suggested names:

```text
checkpoints/l2-baseline/
checkpoints/l2-golden-<steps>/
checkpoints/l2-level2-clear/
demos/l2_golden_<steps>.npz
videos/l2_<milestone>.mp4
```

Keep generated artifacts outside normal Git history. Publish the final checkpoint
and proof video as a release asset, as done for Level 1.

## Conditional recovery tools

Use these only when the simple loop provides evidence that they are needed:

- **PPO consolidation:** deterministic BC works, but broader or stochastic play
  is weak. Preserve the BC checkpoint first and use calm penalties.
- **Demo ensemble:** one golden route does not cover several genuinely different
  entry states. It improves BC coverage but does not by itself prevent PPO
  erosion.
- **Self-imitation refresh:** one corridor erodes during otherwise useful PPO.
  Limit to one or two refresh cycles; long loops oscillated.
- **Save-state curriculum:** a broad recovery skill is tolerant to reset phase.
- **New positional shaping:** video and traces prove a systematic screen-position
  basin. Use a stage-appropriate diagnostic and keep survival as the arbiter.

## Legacy experiments: retain for reference, not as defaults

- broad random frontier search;
- reload-state search used as evidence of full-run transfer;
- local-only BC datasets;
- automatic drill plus two consolidation rounds after every correction;
- high-entropy rescue runs;
- score muting;
- frame-skip 2 as a universal precision fix;
- long interleaved PPO/BC loops;
- horizontal x-position rewards carried blindly into Level 2.

These experiments were useful because they diagnosed failure modes. The canonical
golden-run method supersedes them as the normal advancement loop.

