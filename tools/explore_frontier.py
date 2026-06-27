"""Go-Explore frontier search.

From a save-state (the policy's failure frontier), search for a short action
sequence (a "bridge") that gets the ship past the obstacle AND hands off cleanly to
the current policy. Each candidate is scored by TOTAL survival once control is
returned to the greedy policy after the bridge -- so we find bridges the policy can
actually CONTINUE from, not ones that leave it in an unrecoverable state (the
"derailment" problem). Outputs the bridge as a BC demo + a new frontier save-state.

General by design: scored only by survival steps (stage-clear as primary key), no
wall-specific reward, no "go front at step N".

Usage:
  python -m tools.explore_frontier \
    --model checkpoints/<run>/lifeforce_ppo_<N>_steps.zip \
    --state states/l3_wall.state --name l3_bridge \
    --candidates 1000 --bridge-steps 60 --epsilon 0.5

Then behaviour-clone the demo (tools/self_imitation.py) and resume PPO.
"""
import argparse
import gzip
import os
import time

import numpy as np
from stable_baselines3 import PPO

from src import config as C  # noqa: F401  (kept for parity / future scoring tweaks)
from src.env import make_env


def load_state(path):
    with gzip.open(path, "rb") as fh:
        return fh.read()


def save_state(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def replay(env, frontier, actions):
    """Execute a FIXED action list from `frontier` (no policy, no rng). Deterministic
    -- used only for the determinism sanity check. Returns survival steps."""
    env.unwrapped.initial_state = frontier
    obs, _ = env.reset()
    steps = 0
    for a in actions:
        _, _, term, trunc, _ = env.step(a)
        steps += 1
        if term or trunc:
            break
    return steps


def rollout(env, frontier, model, bridge_steps, macro, eps, cont_cap, rng):
    """Reset to `frontier`, run a `bridge_steps`-long bridge (each macro: with prob
    `eps` a random action, else a sampled policy action, held `macro` decisions),
    then hand to the GREEDY policy for up to `cont_cap` steps. The bridge obs/acts
    are recorded for BC; the emulator state at the handoff is captured as a possible
    new frontier. Score = total survival steps (cleared is the primary key)."""
    env.unwrapped.initial_state = frontier
    obs, info = env.reset()
    b_obs, b_acts = [], []
    steps, cleared, done = 0, False, False

    held, hold_act = 0, None
    for _ in range(bridge_steps):
        if held == 0:
            if rng.random() < eps:
                hold_act = env.action_space.sample()        # real exploration
            else:
                hold_act, _ = model.predict(obs, deterministic=False)
            held = macro
        b_obs.append(np.asarray(obs, dtype=np.uint8))
        b_acts.append(np.asarray(hold_act))
        obs, _, term, trunc, info = env.step(hold_act)
        steps += 1
        held -= 1
        cleared = cleared or bool(info.get("stage_cleared"))
        if term or trunc:
            done = True
            break

    handoff = None if done else env.unwrapped.em.get_state()  # past the bridge, still alive

    c = 0
    while not done and c < cont_cap:
        a, _ = model.predict(obs, deterministic=True)         # handoff to greedy policy
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        c += 1
        cleared = cleared or bool(info.get("stage_cleared"))
        done = term or trunc

    return {"steps": steps, "cleared": cleared, "b_obs": b_obs, "b_acts": b_acts, "handoff": handoff}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", required=True, help="frontier .state (gzipped)")
    p.add_argument("--name", required=True, help="output base name -> demos/<name>.npz, states/<name>.state")
    p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--bridge-steps", type=int, default=60, dest="bridge_steps",
                   help="bridge length in agent-steps; must cover the obstacle maneuver window")
    p.add_argument("--macro", type=int, default=3, help="hold each chosen action this many decisions")
    p.add_argument("--epsilon", type=float, default=0.5,
                   help="prob a bridge macro is RANDOM vs a (stochastic) policy action")
    p.add_argument("--continuation", type=int, default=600, help="greedy-policy handoff step cap")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(preprocess=True, curriculum=False, seed=args.seed)
    env.action_space.seed(args.seed)
    model = PPO.load(args.model, device="cpu")
    frontier = load_state(args.state)

    # --- determinism check: a fixed bridge must replay identically ---
    probe = [env.action_space.sample() for _ in range(30)]
    d1, d2 = replay(env, frontier, probe), replay(env, frontier, probe)
    if d1 != d2:
        print(f"!! WARNING: non-deterministic replay ({d1} vs {d2}) -- Go-Explore assumes determinism!")
    else:
        print(f"determinism OK (fixed bridge survived {d1} steps, reproducible)")

    # --- baseline: greedy policy alone from the frontier (the wall we must beat) ---
    base = rollout(env, frontier, model, 0, args.macro, 0.0, args.continuation, rng)
    baseline = base["steps"]
    print(f"baseline (policy from frontier): {baseline} steps (cleared={base['cleared']})\n")

    # --- search ---
    best = {"steps": -1, "cleared": False}
    t0 = time.perf_counter()
    for i in range(args.candidates):
        r = rollout(env, frontier, model, args.bridge_steps, args.macro, args.epsilon, args.continuation, rng)
        if (r["cleared"], r["steps"]) > (best["cleared"], best["steps"]):
            best = {**r, "bridge": list(r["b_acts"])}
            print(f"  cand {i}: NEW BEST {r['steps']} steps  (+{r['steps']-baseline} vs baseline, cleared={r['cleared']})")
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{args.candidates}  best={best['steps']}  ({time.perf_counter()-t0:.0f}s)")

    gain = best["steps"] - baseline
    print(f"\nbest {best['steps']} steps vs baseline {baseline}  (+{gain}, cleared={best['cleared']})")
    if gain <= 0:
        print("No survivor beyond baseline. Try: more --candidates, longer/shorter --bridge-steps, "
              "higher --epsilon, a different --state (lead time), or CEM.")
        env.close()
        return

    os.makedirs("demos", exist_ok=True)
    demo = f"demos/{args.name}.npz"
    np.savez_compressed(demo,
                        obs=np.asarray(best["b_obs"], dtype=np.uint8),
                        actions=np.asarray(best["b_acts"]))
    print(f"saved demo ({len(best['b_acts'])} bridge steps) -> {demo}")
    if best.get("handoff") is not None:
        st = f"states/{args.name}.state"
        save_state(st, best["handoff"])
        print(f"saved new frontier (past the obstacle) -> {st}")
    else:
        print("(best rollout died within the bridge; no clean handoff state to save)")
    env.close()


if __name__ == "__main__":
    main()
