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
can't be reproduced. The optional --warmup N greedy actions are computed once and
replayed from a fresh reset inside every rollout, never loaded from an intermediate
state. Winners are re-verified with independent resets before anything is saved.

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
import multiprocessing as mp
import os
import time

import numpy as np
from stable_baselines3 import PPO

from src import config as C
from src.env import LifeForceWrapper, make_env

# Default segment move vocabulary (indices into C.MOVES): the commit-through is a
# forward thread whose unknown is the Y-line, so sweep RIGHT-biased moves plus
# pure vertical adjustments. B(fire) is hardwired on in every move.
DEFAULT_MOVES = [4, 6, 8, 1, 2]        # R, UP+R, DOWN+R, UP, DOWN
DEFAULT_DURS = [4, 8, 12, 16]          # agent-steps per segment (frame-skip 4)
GREEDY_MOVE = -1                       # pseudo-move: follow the greedy policy for the
                                       # segment (lets beam search ride the policy where
                                       # it is already right and script only corrections)
ACT_MOVE = -2                          # pseudo-move: HOLD + press A for one step (buy the
                                       # power-up under the meter cursor). One vocab entry,
                                       # so activation costs +1 branching, not 2x.

_WORKER_ENV = None
_WORKER_MODEL = None
_WORKER_START = None
_WORKER_WARMUP = 0
_WORKER_CONTINUATION = 0
_WORKER_WARMUP_ACTIONS = None


def load_state(path):
    with gzip.open(path, "rb") as fh:
        return fh.read()


def save_state(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def rollout(env, start, model, plan, warmup, cont_cap, record=False,
            record_continuation=False, warmup_actions=None):
    """Fresh load of `start` -> deterministic `warmup` greedy steps/actions
    -> scripted segments `plan` = [(move_idx, duration), ...] -> greedy
    continuation. Returns score dict; obs/actions of warmup+segments if record."""
    if start is not None:                         # None = canonical reset (true level
        env.unwrapped.initial_state = start       # start; avoids reload frame-phase
    obs, _ = env.reset()                          # divergence — see RESUME.md)
    d_obs, d_acts = [], []
    steps, max_x, cleared, done = 0, 0, False, False
    terminated, truncated, info = False, False, {}

    rec = {"on": record}                          # demo = warmup+segments ONLY, not
                                                  # the greedy continuation (it would
                                                  # dilute the BC signal ~5:1)
    def _step(act):
        nonlocal obs, steps, max_x, cleared, done, terminated, truncated, info
        if rec["on"]:
            d_obs.append(np.asarray(obs, dtype=np.uint8))
            d_acts.append(np.asarray(act))
        obs, _, terminated, truncated, info = env.step(act)
        steps += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        cleared = cleared or bool(info.get("stage_cleared"))
        done = terminated or truncated

    # Cached-action search does not need resized/stacked observations for the
    # old prefix. Step the gameplay/reward wrapper directly, then run the final
    # FRAME_STACK actions through the full observation pipeline to rebuild the
    # exact current policy input. Recording still uses the full pipeline because
    # every observation belongs in a golden demo.
    gameplay_env = env
    while not isinstance(gameplay_env, LifeForceWrapper):
        gameplay_env = gameplay_env.env

    def _fast_step(act):
        nonlocal steps, max_x, cleared, done, terminated, truncated, info
        _, _, terminated, truncated, info = gameplay_env.step(act)
        steps += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        cleared = cleared or bool(info.get("stage_cleared"))
        done = terminated or truncated

    if warmup_actions is not None and len(warmup_actions) != warmup:
        raise ValueError(f"cached warmup has {len(warmup_actions)} actions, expected {warmup}")
    fast_until = 0
    if warmup_actions is not None and not record:
        fast_until = max(0, warmup - C.FRAME_STACK)
    for i in range(warmup):                       # deterministic canonical run-up
        if done:
            break
        if warmup_actions is None:
            a, _ = model.predict(obs, deterministic=True)
        else:
            # Replaying the policy's previously computed deterministic actions
            # preserves the canonical reset world while avoiding thousands of
            # redundant CNN inferences for every search candidate.
            a = warmup_actions[i]
        if i < fast_until:
            _fast_step(a)
        else:
            _step(a)

    seg_end_state = None
    for move, dur in plan:                        # the scripted segments
        for _ in range(dur):
            if done:
                break
            if move == GREEDY_MOVE:               # pseudo-move: follow the policy for
                a, _ = model.predict(obs, deterministic=True)  # d steps (deterministic,
                a = np.asarray(a).ravel()         # recomputed per replay -> reproducible)
                # suppress the policy's OWN activate presses inside GREEDY segments:
                # ride its movement (chase/kill/collect) but never let it spend the
                # meter — all purchases go through deliberate ACT segments.
                _step(np.array([int(a[0]), 0]))
            elif move == ACT_MOVE:                # pseudo-move: HOLD + press A (activate)
                _step(np.array([0, 1]))
            else:
                _step(np.array([move, 0]))
        if done:
            break
    if not done:
        seg_end_state = env.unwrapped.em.get_state()
    # A stage clear during the script is success, not an in-script failure.
    survived_segments = not done or cleared
    # loadout AT THE HANDOFF (end of script): this is what the script achieved.
    # Measured here, not at episode end — the greedy continuation may waste the
    # banked capsules (e.g. A-mashing buys Speed), which would erase the ranking
    # signal for plans that farm correctly.
    ram = env.unwrapped.get_ram()
    loadout = {"powerbar": int(ram[C.ADDR_POWERBAR]), "missile": int(ram[C.ADDR_MISSILE]),
               "options": int(ram[C.ADDR_OPTIONS]), "shield": int(ram[C.ADDR_SHIELD]),
               "speed": int(ram[C.ADDR_SPEED])}

    # Normal search demos contain only the warmup+script so a long continuation
    # does not dilute the new maneuver.  Golden-run demos deliberately retain the
    # whole continuous trajectory: familiar greedy frames protect old skills while
    # the scripted correction is cloned in its true, warm-frame-stack context.
    rec["on"] = bool(record and record_continuation)
    c = 0
    while not done and c < cont_cap:              # clean-handoff scoring
        a, _ = model.predict(obs, deterministic=True)
        _step(a)
        c += 1

    # pu scoring (all three rules measured, not theoretical): speed weighs 0 —
    # it's the cheapest slot and a loadout-ranked beam farms it greedily,
    # burning capsules needed for Missile/Options. Powerbar weighs +1 as a
    # PROGRESS BREADCRUMB — without it, an eaten-but-unspent capsule scores
    # nothing and the search must find a complete eat->eat->ACT chain blind.
    # Purchase weights are set so completing a purchase strictly beats hoarding:
    # missile 5 > 2 banked; option 8 > 5 banked; shield 7 > 6 banked.
    pu = (loadout["powerbar"] + 5 * loadout["missile"] + 8 * loadout["options"]
          + 7 * loadout["shield"])

    return {"steps": steps, "max_x": max_x, "cleared": cleared,
            "terminated": terminated, "truncated": truncated,
            "life_lost": bool(info.get("life_lost")),
            "survived_segments": survived_segments, "handoff": seg_end_state,
            "obs": d_obs, "acts": d_acts, "plan": plan,
            "loadout": loadout, "pu": pu}


def _init_rollout_worker(model_path, start, seed, warmup, continuation,
                         warmup_actions):
    """Initialize one persistent emulator/model pair for parallel beam scoring."""
    global _WORKER_ENV, _WORKER_MODEL, _WORKER_START, _WORKER_WARMUP
    global _WORKER_CONTINUATION, _WORKER_WARMUP_ACTIONS
    _WORKER_ENV = make_env(preprocess=True, curriculum=False, seed=seed)
    _WORKER_MODEL = PPO.load(model_path, device="cpu")
    _WORKER_START = start
    _WORKER_WARMUP = warmup
    _WORKER_CONTINUATION = continuation
    _WORKER_WARMUP_ACTIONS = warmup_actions


def _worker_rollout(plan):
    return rollout(
        _WORKER_ENV, _WORKER_START, _WORKER_MODEL, plan,
        _WORKER_WARMUP, _WORKER_CONTINUATION,
        warmup_actions=_WORKER_WARMUP_ACTIONS,
    )


def beam_search(env, start, model, args, base, warmup_actions=None):
    """Grow the scripted plan segment-by-segment with beam pruning. Needed when the
    dangerous window is LONGER than an enumerable plan: from x120 the camping death
    is at ~step 107, and any short script that hands off early gets erased by the
    policy's retreat prior (measured: script to x=211, greedy retreats to x=15 by
    step 65, dies at 107 — identical to baseline). So the script must survive PAST
    the danger before handing off. Deterministic env -> this is planning; score each
    prefix by TOTAL survival (script + greedy continuation) and keep the top-K
    surviving prefixes per round. Returns the accepted rollout or None."""
    rng = np.random.default_rng(args.seed)
    moves = list(args.moves) + ([GREEDY_MOVE] if args.greedy_move else [])
    vocab = list(itertools.product(moves, args.durations))
    if args.activate:
        vocab.append((ACT_MOVE, 1))                # single extra entry: press A once
    pu_mode = args.rank_powerups

    def key(r, cont):
        # pu-mode: loadout first (that's the objective), then survival, then max_x.
        # normal: total survival first (steps = level progress in an auto-scroller).
        if pu_mode:
            return (r["cleared"], r["pu"], r["steps"], r["max_x"])
        return (r["cleared"], r["steps"], r["max_x"])

    pool = None
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            args.workers,
            initializer=_init_rollout_worker,
            initargs=(args.model, start, args.seed, args.warmup,
                      args.continuation, warmup_actions),
        )
        print(f"parallel candidate scoring: {args.workers} workers")

    beams = [[]]                                   # surviving prefixes (round 0: empty)
    best, t0 = None, time.perf_counter()
    try:
        for rnd in range(args.rounds):
            scored = []
            plans = [prefix + [(m, d)] for prefix in beams for m, d in vocab]
            if pool is None:
                results = [
                    rollout(env, start, model, plan, args.warmup, args.continuation,
                            warmup_actions=warmup_actions)
                    for plan in plans
                ]
            else:
                results = pool.map(_worker_rollout, plans, chunksize=1)
            for plan, r in zip(plans, results):
                script_len = sum(dd for _, dd in plan)
                if not r["survived_segments"]:
                    continue                        # died in-script: prune this branch
                cont = r["steps"] - script_len - args.warmup
                scored.append((key(r, cont), rng.random(), plan, r, cont))
            if not scored:
                print(f"  round {rnd+1}: every extension died in-script — beam exhausted")
                break
            scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
            top = scored[0]
            if best is None or top[0] > best[0]:
                best = top
            beams = [s[2] for s in scored[:args.beam]]
            el = time.perf_counter() - t0
            tr = top[3]
            pu_str = f" pu={tr['pu']} {tr['loadout']}" if pu_mode else ""
            print(f"  round {rnd+1}/{args.rounds}: {len(scored)} survivors, best total "
                  f"{tr['steps']} steps (+{tr['steps']-base['steps']}) max_x={tr['max_x']} "
                  f"cont={top[4]}{pu_str}  [{fmt_plan(top[2])}]  ({el:.0f}s)")
            # accept early once the handoff itself is healthy: script survived AND the
            # greedy continuation lives long past the danger window (pu-mode: also
            # require an actual loadout gain — that's the point of the search)
            gained = ((tr["pu"] > base["pu"]) if pu_mode
                      else (tr["steps"] - base["steps"] >= args.min_gain))
            if top[4] >= args.accept_cont and gained:
                print(f"  continuation {top[4]} >= {args.accept_cont}: accepting")
                best = top
                break
            if sum(d for _, d in beams[0]) + args.warmup >= args.script_cap:
                print(f"  script length reached --script-cap {args.script_cap}")
                break
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if best is None:
        return None
    br = best[3]
    if pu_mode:
        # accept on loadout gain with survival not collapsing vs baseline
        if br["pu"] <= base["pu"] or br["steps"] < base["steps"] - args.pu_step_slack:
            return None
    elif br["steps"] - base["steps"] < args.min_gain:
        return None
    # verify: 2 independent fresh reloads must reproduce, then record the demo
    plan = best[2]
    v1 = rollout(env, start, model, plan, args.warmup, args.continuation,
                 warmup_actions=warmup_actions)
    v2 = rollout(env, start, model, plan, args.warmup, args.continuation, record=True,
                 record_continuation=args.record_continuation,
                 warmup_actions=warmup_actions)
    ok = {(v["steps"], v["max_x"], v["pu"]) for v in (v1, v2)} == {(br["steps"], br["max_x"], br["pu"])}
    verdict = ("REPRODUCED 2/2" if ok
               else f"MISMATCH ({v1['steps']}/{v2['steps']} vs {br['steps']})")
    print(f"verify best [{fmt_plan(plan)}]: {verdict}  pu={br['pu']} loadout={br['loadout']}")
    return v2 if ok else None


def fmt_plan(plan):
    names = {1: "UP", 2: "DOWN", 4: "R", 6: "UP+R", 8: "DOWN+R", 0: "HOLD",
             3: "LEFT", 5: "UP+L", 7: "DOWN+L", GREEDY_MOVE: "GREEDY",
             ACT_MOVE: "ACT"}
    return " > ".join(f"{names.get(m, m)}x{d}" for m, d in plan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", required=True,
                   help="RESET for canonical env.reset()+--warmup, or a gzipped "
                        ".state for diagnostic reload-world search")
    p.add_argument("--name", required=True, help="output base -> demos/<name>.npz, states/<name>.state")
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--moves", type=int, nargs="+", default=DEFAULT_MOVES,
                   help=f"C.MOVES indices allowed per segment (default {DEFAULT_MOVES})")
    p.add_argument("--durations", type=int, nargs="+", default=DEFAULT_DURS,
                   help=f"segment lengths in agent-steps (default {DEFAULT_DURS})")
    p.add_argument("--warmup", type=int, default=0,
                   help="greedy steps before the segments; actions are computed once "
                        "and replayed from reset (shifts the search window without "
                        "capturing a frame-phase-shifted state)")
    p.add_argument("--continuation", type=int, default=600, help="greedy handoff step cap")
    p.add_argument("--min-gain", type=int, default=30, dest="min_gain",
                   help="accept only if verified steps beat the greedy baseline by this many")
    p.add_argument("--top", type=int, default=5, help="verify this many top candidates")
    p.add_argument("--max-candidates", type=int, default=0, dest="max_candidates",
                   help="0 = exhaustive; else uniform subsample of the plan space")
    p.add_argument("--beam", type=int, default=0,
                   help="beam width; >0 switches to segment-by-segment beam search "
                        "(for maneuvers longer than an enumerable plan)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel emulator workers for beam candidate scoring")
    p.add_argument("--greedy-move", action="store_true", dest="greedy_move",
                   help="add a GREEDY pseudo-move to the vocab: a segment that follows "
                        "the policy (recorded; deterministic). Beam only.")
    p.add_argument("--activate", action="store_true",
                   help="add an ACT pseudo-move to the vocab: HOLD + press A for one "
                        "step (buy the power-up under the meter cursor). Beam only.")
    p.add_argument("--rank-powerups", action="store_true", dest="rank_powerups",
                   help="beam ranks plans by LOADOUT first (4*missile+3*options+"
                        "2*shield+speed), survival second — for farming openings. "
                        "Acceptance: pu gain with steps >= baseline - --pu-step-slack.")
    p.add_argument("--pu-step-slack", type=int, default=50, dest="pu_step_slack",
                   help="pu-mode: how many survival steps a loadout plan may give up "
                        "vs the greedy baseline and still be accepted")
    p.add_argument("--rounds", type=int, default=20, help="max beam rounds (segments)")
    p.add_argument("--script-cap", type=int, default=200, dest="script_cap",
                   help="stop growing the script past this many agent-steps")
    p.add_argument("--accept-cont", type=int, default=150, dest="accept_cont",
                   help="accept once the greedy continuation alone survives this long")
    p.add_argument("--record-continuation", action="store_true", dest="record_continuation",
                   help="include the greedy continuation in the saved demo (canonical "
                        "golden-run BC; normally only warmup+script are recorded)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be at least 1")

    env = make_env(preprocess=True, curriculum=False, seed=args.seed)
    model = PPO.load(args.model, device="cpu")
    # RESET = search the canonical-reset world (true level start). Any saved-state
    # reload lives in a frame-phase-shifted world whose scripts do NOT transfer to
    # real runs (measured: reload scripts die on the continuous trajectory).
    start = None if args.state == "RESET" else load_state(args.state)

    # Compute a deterministic policy run-up once, then replay the exact actions
    # for every candidate. This keeps every rollout in the canonical reset world
    # (unlike save-state reloads) while making deep-frontier searches practical.
    warmup_actions = None
    if args.warmup:
        warm = rollout(env, start, model, [], args.warmup, 0, record=True)
        if warm["steps"] != args.warmup:
            raise SystemExit(
                f"!! policy ended after {warm['steps']} steps during --warmup {args.warmup}"
            )
        warmup_actions = np.asarray(warm["acts"])
        replay = rollout(env, start, model, [], args.warmup, 0,
                         warmup_actions=warmup_actions)
        replay_keys = ("steps", "max_x", "pu", "cleared")
        if any(replay[k] != warm[k] for k in replay_keys):
            raise SystemExit("!! cached warmup replay diverged from the policy run-up")
        print(f"cached {len(warmup_actions)} deterministic warmup actions")

    # --- determinism probe: a fixed plan must replay identically from fresh loads ---
    probe = [(4, 8), (2, 8), (6, 8)]
    r1 = rollout(env, start, model, probe, args.warmup, args.continuation,
                 warmup_actions=warmup_actions)
    r2 = rollout(env, start, model, probe, args.warmup, args.continuation,
                 warmup_actions=warmup_actions)
    if (r1["steps"], r1["max_x"]) != (r2["steps"], r2["max_x"]):
        raise SystemExit(f"!! non-deterministic replay ({r1['steps']}/{r1['max_x']} vs "
                         f"{r2['steps']}/{r2['max_x']}) -- aborting, results would be meaningless")
    print(f"determinism OK (probe plan: {r1['steps']} steps / max_x {r1['max_x']}, reproducible)")

    # --- greedy baseline from the same start (the bar to beat) ---
    base = rollout(env, start, model, [], args.warmup, args.continuation,
                   warmup_actions=warmup_actions)
    print(f"greedy baseline: {base['steps']} steps / max_x {base['max_x']} / "
          f"cleared={base['cleared']} / pu={base['pu']} {base['loadout']}\n")
    if base["truncated"] and not base["cleared"]:
        raise SystemExit(
            f"!! greedy baseline hit MAX_EPISODE_STEPS={C.MAX_EPISODE_STEPS}; "
            "raise the ceiling before searching or no candidate can establish its true gain"
        )

    # --- beam mode: grow the script past the danger window, then verify+save ---
    if args.beam > 0:
        print(f"beam search: width={args.beam} vocab={len(args.moves)}x{len(args.durations)} "
              f"rounds<={args.rounds} script-cap={args.script_cap} accept-cont={args.accept_cont}")
        accepted = beam_search(env, start, model, args, base, warmup_actions)
        if accepted is None:
            print(f"\nNo verified beam plan beat baseline by {args.min_gain}+ steps.")
            print("Signal: the corridor may need finer moves (--durations 2 4 8), a different "
                  "start state, or closed-loop control -> human demo fallback.")
        else:
            _save_outputs(accepted, base, args)
        env.close()
        return

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
        r = rollout(env, start, model, list(plan), args.warmup, args.continuation,
                    warmup_actions=warmup_actions)
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
        v1 = rollout(env, start, model, list(r["plan"]), args.warmup, args.continuation,
                     warmup_actions=warmup_actions)
        v2 = rollout(env, start, model, list(r["plan"]), args.warmup, args.continuation,
                     record=True, record_continuation=args.record_continuation,
                     warmup_actions=warmup_actions)
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

    _save_outputs(accepted, base, args)
    env.close()


def _save_outputs(accepted, base, args):
    os.makedirs("demos", exist_ok=True)
    demo = f"demos/{args.name}.npz"
    np.savez_compressed(demo, obs=np.asarray(accepted["obs"], dtype=np.uint8),
                        actions=np.asarray(accepted["acts"]))
    st = None
    if accepted["handoff"] is not None:
        st = f"states/{args.name}.state"
        save_state(st, accepted["handoff"])
    print(f"\nACCEPTED: {fmt_plan(accepted['plan'])}")
    print(f"  loadout: pu={accepted['pu']} {accepted['loadout']}")
    print(f"  {accepted['steps']} steps (+{accepted['steps']-base['steps']} vs baseline) "
          f"max_x={accepted['max_x']} cleared={accepted['cleared']}")
    demo_scope = "full golden trajectory" if args.record_continuation else "warmup+segments"
    print(f"  demo ({len(accepted['acts'])} steps: {demo_scope}) -> {demo}")
    if st is not None:
        print(f"  handoff state (segment end) -> {st}")
    else:
        print("  handoff state: none (the scripted segment cleared the level)")
    print(f"next: python -m tools.self_imitation --model {args.model} --demos {demo} "
          f"--out checkpoints/l3-bc/lifeforce_ppo_bc.zip")


if __name__ == "__main__":
    main()
