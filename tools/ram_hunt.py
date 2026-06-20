"""RAM-address hunt for Life Force (NES).

The bundled stable-retro integration only maps `score` and `lives`. To detect
"Level 1 cleared" we need the RAM address of the stage/level counter, which is
undocumented. This tool helps find it by the classic memory-scanning method.

Strategy implemented here (baseline pass):
  - Capture the full 2 KB RAM repeatedly across a Stage-1 rollout.
  - The stage counter must be CONSTANT throughout a single stage, and hold a
    SMALL value (Life Force has 6 stages -> roughly 0..6).
  - Report the bytes matching those criteria as candidates, and save the
    baseline snapshot so we can later diff it against a Stage-2 reference
    (the counter is whichever candidate flips from its Stage-1 value to +1).

Usage:
  python tools/ram_hunt.py baseline           # capture + analyze Stage 1
  python tools/ram_hunt.py baseline --steps 800 --out ram_dumps/stage1.npz
"""
import argparse
import os

import numpy as np
import stable_retro as retro

GAME = "LifeForce-Nes-v0"
DEFAULT_STATE = "1Player.Level1"

# Known variables from the bundled data.json (exclude from candidate lists).
KNOWN = {
    52: "lives (u1)",
    2020: "score byte0 (d4)", 2021: "score byte1", 2022: "score byte2", 2023: "score byte3",
}


def safe_close(env):
    """env.close() can raise a cosmetic pyglet/Cocoa teardown error on macOS."""
    try:
        env.close()
    except Exception:
        pass


def collect_trace(state, steps, seed=0):
    """Run a rollout and return a (T, 2048) array of RAM snapshots + the frames count.

    Uses a fixed "hold fire + drift right" action so we actually move through the
    stage rather than sitting still. Life Force is MultiBinary(9); on NES the
    buttons are [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]. We hold B
    (fire) and RIGHT to make progress.
    """
    env = retro.make(GAME, state=state)
    env.reset(seed=seed)
    n = env.action_space.n  # MultiBinary(9) -> 9
    act = np.zeros(n, dtype=np.int8)
    act[0] = 1   # B = fire
    act[7] = 1   # RIGHT = move forward

    rams = [env.get_ram().copy()]
    info_last = {}
    for t in range(steps):
        _, _, term, trunc, info = env.step(act)
        rams.append(env.get_ram().copy())
        info_last = info
        if term or trunc:
            print(f"  episode ended at step {t} (term={term} trunc={trunc})")
            break
    safe_close(env)
    return np.array(rams), info_last


def analyze_baseline(trace):
    """Surface the two kinds of address we care about.

    1. stage-counter candidates: constant across the whole stage AND a small
       NON-ZERO value (a 1-indexed stage reads 1..6). Excluding zero drops the
       ~1000 unused/zero bytes that otherwise drown the signal.
    2. progress candidates: bytes that monotonically (non-decreasingly) increase
       as the stage auto-scrolls -> a dense "how far into the stage" signal.
    """
    first, last = trace[0], trace[-1]
    constant = (trace == first).all(axis=0)

    stage_candidates = [
        a for a in np.where(constant)[0]
        if 1 <= int(first[a]) <= 6 and a not in KNOWN
    ]

    # Monotonic non-decreasing AND actually grew over the rollout. Ignore bytes
    # that wrap (a fast frame-timer wraps every 256 frames); require the net
    # change to be positive and the series to never decrease.
    diffs = np.diff(trace.astype(np.int16), axis=0)
    non_decreasing = (diffs >= 0).all(axis=0)
    grew = (last.astype(np.int16) - first.astype(np.int16)) > 0
    progress_candidates = [
        a for a in np.where(non_decreasing & grew)[0] if a not in KNOWN
    ]
    return constant, stage_candidates, progress_candidates, first, last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["baseline"])
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", default="ram_dumps/stage1_baseline.npz")
    args = p.parse_args()

    print(f"Collecting Stage-1 RAM trace: state={args.state} steps={args.steps}")
    trace, info = collect_trace(args.state, args.steps)
    print(f"  captured {trace.shape[0]} snapshots of {trace.shape[1]} bytes")
    print(f"  final info: {info}")

    constant, stage_c, progress_c, first, last = analyze_baseline(trace)
    print(f"\nConstant bytes across rollout: {int(constant.sum())} / 2048")

    print(f"\nSTAGE-COUNTER candidates (constant, value 1..6): {len(stage_c)}")
    print("  addr   hex     value")
    for a in stage_c:
        print(f"  {a:4d}   0x{a:03x}   {int(first[a])}")

    print(f"\nPROGRESS candidates (monotonically increasing): {len(progress_c)}")
    print("  addr   hex     first -> last")
    for a in progress_c:
        print(f"  {a:4d}   0x{a:03x}   {int(first[a])} -> {int(last[a])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, trace=trace, first=first, last=last,
                        constant=constant, stage_candidates=np.array(stage_c),
                        progress_candidates=np.array(progress_c))
    print(f"\nSaved baseline -> {args.out}")
    print("Next: capture a Stage-2 reference and diff to pin the stage counter; "
          "watch the progress candidates live to confirm scroll position.")


if __name__ == "__main__":
    main()
