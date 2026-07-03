"""Structured segment search for a reproducible commit-through demo.

Purpose (see docs/go-explore-l3-progress.md, Session 6): the L3 pinch needs a
narrow, precisely-timed sustained maneuver that per-step random search cannot
assemble (13k random bridges plateaued) and the current policy actively avoids
(it retreats at the window). Instead of random actions, exhaustively sweep a SMALL
STRUCTURED space of segment plans -- N consecutive holds, each (move, duration) --
from a fresh save-state load, and score each plan by GREEDY-POLICY CONTINUATION
survival after the scripted segments end (clean-handoff criterion, same as
explore_frontier).

Reproducibility by construction (the failure that invalidated the earlier
Go-Explore run): every candidate starts from a FRESH load of the saved state --
there is no greedy advance-to-near-failure phase whose accumulated emulator state
can't be reproduced. The optional --warmup N greedy steps are RE-COMPUTED inside
every rollout (deterministic policy from a deterministic reset), never captured
as an intermediate state. Winners are re-verified with independent reloads before
anything is saved.

Usage:
  python -m tools.segment_search \
    --model checkpoints/l3-pinch-mixed-long/lifeforce_ppo_700000_steps.zip \
    --state states/l3_pinch_curriculum/l3_pinch_x120.state \
    --name l3_thread_x120

Outputs (only if a verified candidate beats the greedy baseline by --min-gain):
  demos/<name>.npz      -- BC demo: obs+actions of the scripted segments (+warmup)
  states/<name>.state   -- emulator state at segment end (clean-handoff frontier)
"""
import argparse
import gzip
import itertools
import os
import time

import numpy as np
from stable_baselines3 import PPO

from src import config as C
from src.env import make_env

# Default segment move vocabulary (indices into C.MOVES): the commit-through is a
# forward thread whose unknown is the Y-line, so sweep RIGHT-biased moves plus
# pure vertical adjustments. B(fire) is hardwired on in every move.
DEFAULT_MOVES = [4, 6, 8, 1, 2]        # R, UP+R, DOWN+R, UP, DOWN
DEFAULT_DURS = [4, 8, 12, 16]          # agent-steps per segment (frame-skip 4)


def load_state(path):
    with gzip.open(path, "rb") as fh:
        return fh.read()


def save_state(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def rollout(env, start, model, plan, warmup, cont_cap, record=False):
    """Fresh load of `start` -> `warmup` greedy steps (recomputed, deterministic)
    -> scripted segments `plan` = [(move_idx, duration), ...] -> greedy
    continuation. Returns score dict; obs/actions of warmup+segments if record."""
    env.unwrapped.initial_state = start
    obs, _ = env.reset()
    d_obs, d_acts = [], []
    steps, max_x, cleared, done = 0, 0, False, False

    def _step(act):
        nonlocal obs, steps, max_x, cleared, done
        if record:
            d_obs.append(np.asarray(obs, dtype=np.uint8))
            d_acts.append(np.asarray(act))
        obs, _, term, trunc, info = env.step(act)
        steps += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        cleared = cleared or bool(info.get("stage_cleared"))
        done = term or trunc

    for _ in range(warmup):                       # deterministic run-up, re-computed
        if done:
            break
        a, _ = model.predict(obs, deterministic=True)
        _step(a)

    seg_end_state = None
    for move, dur in plan:                        # the scripted segments
        for _ in range(dur):
            if done:
                break
            _step(np.array([move, 0]))
        if done:
            break
    if not done:
        seg_end_state = env.unwrapped.em.get_state()
    survived_segments = not done

    c = 0
    while not done and c < cont_cap:              # clean-handoff scoring
        a, _ = model.predict(obs, deterministic=True)
        _step(a)
        c += 1

    return {"steps": steps, "max_x": max_x, "cleared": cleared,
            "survived_segments": survived_segments, "handoff": seg_end_state,
            "obs": d_obs, "acts": d_acts, "plan": plan}


def fmt_plan(plan):
    names = {1: "UP", 2: "DOWN", 4: "R", 6: "UP+R", 8: "DOWN+R", 0: "HOLD",
             3: "LEFT", 5: "UP+L", 7: "DOWN+L"}
    return " > ".join(f"{names.get(m, m)}x{d}" for m, d in plan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", required=True, help="start .state (gzipped); loaded FRESH per candidate")
    p.add_argument("--name", required=True, help="output base -> demos/<name>.npz, states/<name>.state")
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--moves", type=int, nargs="+", default=DEFAULT_MOVES,
                   help=f"C.MOVES indices allowed per segment (default {DEFAULT_MOVES})")
    p.add_argument("--durations", type=int, nargs="+", default=DEFAULT_DURS,
                   help=f"segment lengths in agent-steps (default {DEFAULT_DURS})")
    p.add_argument("--warmup", type=int, default=0,
                   help="greedy steps before the segments, RE-COMPUTED every rollout "
                        "(shifts the search window without capturing a new state)")
    p.add_argument("--continuation", type=int, default=600, help="greedy handoff step cap")
    p.add_argument("--min-gain", type=int, default=30, dest="min_gain",
                   help="accept only if verified steps beat the greedy baseline by this many")
    p.add_argument("--top", type=int, default=5, help="verify this many top candidates")
    p.add_argument("--max-candidates", type=int, default=0, dest="max_candidates",
                   help="0 = exhaustive; else uniform subsample of the plan space")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env(preprocess=True, curriculum=False, seed=args.seed)
    model = PPO.load(args.model, device="cpu")
    start = load_state(args.state)

    # --- determinism probe: a fixed plan must replay identically from fresh loads ---
    probe = [(4, 8), (2, 8), (6, 8)]
    r1 = rollout(env, start, model, probe, args.warmup, args.continuation)
    r2 = rollout(env, start, model, probe, args.warmup, args.continuation)
    if (r1["steps"], r1["max_x"]) != (r2["steps"], r2["max_x"]):
        raise SystemExit(f"!! non-deterministic replay ({r1['steps']}/{r1['max_x']} vs "
                         f"{r2['steps']}/{r2['max_x']}) -- aborting, results would be meaningless")
    print(f"determinism OK (probe plan: {r1['steps']} steps / max_x {r1['max_x']}, reproducible)")

    # --- greedy baseline from the same start (the bar to beat) ---
    base = rollout(env, start, model, [], args.warmup, args.continuation)
    print(f"greedy baseline: {base['steps']} steps / max_x {base['max_x']} / cleared={base['cleared']}\n")

    # --- exhaustive sweep of the plan space ---
    space = list(itertools.product(
        *[list(itertools.product(args.moves, args.durations))] * args.segments))
    rng = np.random.default_rng(args.seed)
    if args.max_candidates and len(space) > args.max_candidates:
        idx = rng.choice(len(space), size=args.max_candidates, replace=False)
        space = [space[i] for i in idx]
        print(f"subsampled {len(space)} of the plan space")
    print(f"sweeping {len(space)} plans "
          f"({args.segments} segments x {len(args.moves)} moves x {len(args.durations)} durations)")

    results, t0 = [], time.perf_counter()
    best_key = (False, base["steps"], base["max_x"])
    for i, plan in enumerate(space):
        r = rollout(env, start, model, list(plan), args.warmup, args.continuation)
        results.append(r)
        key = (r["cleared"], r["steps"], r["max_x"])
        if key > best_key:
            best_key = key
            print(f"  [{i+1}/{len(space)}] NEW BEST {r['steps']} steps (+{r['steps']-base['steps']}) "
                  f"max_x={r['max_x']} handoff={'yes' if r['survived_segments'] else 'DIED IN SEGMENTS'}  "
                  f"{fmt_plan(plan)}")
        if (i + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  ... {i+1}/{len(space)}  best={best_key[1]} steps  "
                  f"({el:.0f}s, {el/(i+1)*1000:.0f}ms/plan)")

    results.sort(key=lambda r: (r["cleared"], r["steps"], r["max_x"]), reverse=True)
    print(f"\ntop {min(args.top, 20)} of {len(results)} (baseline {base['steps']} steps / max_x {base['max_x']}):")
    for r in results[:min(args.top, 20)]:
        print(f"  {r['steps']:4d} steps  max_x={r['max_x']:3d}  cleared={r['cleared']}  "
              f"handoff={'y' if r['survived_segments'] else 'n'}  {fmt_plan(r['plan'])}")

    # --- verify winners: 2 independent fresh reloads must reproduce exactly ---
    accepted = None
    for r in results[:args.top]:
        if r["steps"] - base["steps"] < args.min_gain or not r["survived_segments"]:
            continue
        v1 = rollout(env, start, model, list(r["plan"]), args.warmup, args.continuation)
        v2 = rollout(env, start, model, list(r["plan"]), args.warmup, args.continuation, record=True)
        ok = ({(v["steps"], v["max_x"]) for v in (v1, v2)} == {(r["steps"], r["max_x"])})
        verdict = ("REPRODUCED 2/2" if ok
                   else f"MISMATCH ({v1['steps']}/{v2['steps']} vs {r['steps']})")
        print(f"verify {fmt_plan(r['plan'])}: {verdict}")
        if ok:
            accepted = v2
            break

    if accepted is None:
        print(f"\nNo verified plan beat baseline by {args.min_gain}+ steps with a clean handoff.")
        print("Signal: if top plans die IN segments near the pinch, the maneuver likely needs "
              "closed-loop control -> record a human demo instead (same BC pipeline).")
        env.close()
        return

    os.makedirs("demos", exist_ok=True)
    demo = f"demos/{args.name}.npz"
    np.savez_compressed(demo, obs=np.asarray(accepted["obs"], dtype=np.uint8),
                        actions=np.asarray(accepted["acts"]))
    st = f"states/{args.name}.state"
    save_state(st, accepted["handoff"])
    print(f"\nACCEPTED: {fmt_plan(accepted['plan'])}")
    print(f"  {accepted['steps']} steps (+{accepted['steps']-base['steps']} vs baseline) "
          f"max_x={accepted['max_x']} cleared={accepted['cleared']}")
    print(f"  demo ({len(accepted['acts'])} steps: warmup+segments) -> {demo}")
    print(f"  handoff state (segment end) -> {st}")
    print(f"next: python -m tools.self_imitation --model {args.model} --demos {demo} "
          f"--out checkpoints/l3-bc/lifeforce_ppo_bc.zip")
    env.close()


if __name__ == "__main__":
    main()
