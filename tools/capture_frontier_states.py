"""Capture deterministic frontier/approach states from a trained policy.

This is for the "policy reaches a new failure point, then retreats/dies" loop:
run a checkpoint from a known saved state, inspect the deterministic trajectory,
and save emulator states at reproducible lead-ins such as 50/30/15 steps before
death or the first time the ship reaches selected screen-x thresholds.

Unlike tools/explore_frontier.py, this does not search or claim a pass. It only
captures clean curriculum starts for the next PPO drill.

Usage:
  python -m tools.capture_frontier_states \
    --model checkpoints/l3-wall-hold2/lifeforce_ppo_final.zip \
    --from-state states/l3_wall.state --prefix l3_pinch \
    --before-death 50 30 15 --x-targets 100 130 150
"""
import argparse
import gzip
import os
from dataclasses import dataclass

from stable_baselines3 import PPO

from src import config as C
from src.env import make_env


@dataclass
class Snapshot:
    step: int
    x: int
    y: int
    state: bytes


def load_state(path):
    with gzip.open(path, "rb") as fh:
        return fh.read()


def save_state(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def run_episode(model_path, from_state, deterministic=True, seed=0, frame_skip=None):
    env = make_env(render_mode=None, preprocess=True, curriculum=False, seed=seed,
                   frame_skip=frame_skip)
    if from_state:
        env.unwrapped.initial_state = load_state(from_state)
    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset(seed=seed)
    em = env.unwrapped.em
    trace = []
    done = False
    info = {}

    while not done:
        ram = env.unwrapped.get_ram()
        trace.append(Snapshot(
            step=len(trace),
            x=int(ram[C.ADDR_X_POS]),
            y=int(ram[C.ADDR_Y_POS]),
            state=em.get_state(),
        ))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

    ram = env.unwrapped.get_ram()
    terminal = Snapshot(
        step=len(trace),
        x=int(ram[C.ADDR_X_POS]),
        y=int(ram[C.ADDR_Y_POS]),
        state=em.get_state(),
    )
    try:
        env.close()
    except Exception:
        pass
    return trace, terminal, info


def first_at_or_above(trace, x_target):
    for snap in trace:
        if snap.x >= x_target:
            return snap
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint driving the deterministic capture")
    p.add_argument("--from-state", default=None, dest="from_state",
                   help="optional saved .state to start from, e.g. states/l3_wall.state")
    p.add_argument("--prefix", required=True,
                   help="output prefix -> states/<prefix>_<kind>.state")
    p.add_argument("--before-death", nargs="*", type=int, default=[],
                   help="save these many agent-steps before death/truncation")
    p.add_argument("--x-targets", nargs="*", type=int, default=[],
                   help="save first state where x_pos >= each target")
    p.add_argument("--sample", action="store_true",
                   help="sample actions instead of deterministic greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=None, dest="frame_skip",
                   help=f"override env frame-skip for capture (default config={C.FRAME_SKIP})")
    args = p.parse_args()

    trace, terminal, info = run_episode(
        args.model, args.from_state, deterministic=not args.sample,
        seed=args.seed, frame_skip=args.frame_skip)
    if not trace:
        raise SystemExit("episode ended before any pre-step state could be captured")

    max_snap = max(trace, key=lambda s: s.x)
    mean_x = sum(s.x for s in trace) / len(trace)
    print(f"episode: steps={terminal.step} max_x={max_snap.x} at step={max_snap.step} "
          f"terminal_x={terminal.x} mean_x={mean_x:.1f} score={info.get('score')}")
    print(f"terminal info: max_x={info.get('max_x')} terminal_x={info.get('terminal_x')} "
          f"mean_x={info.get('mean_x')}")

    os.makedirs(C.CURRICULUM_DIR, exist_ok=True)
    saved = []

    for lead in args.before_death:
        idx = terminal.step - lead
        if idx < 0 or idx >= len(trace):
            print(f"skip before-death {lead}: step {idx} outside trace")
            continue
        snap = trace[idx]
        out = os.path.join(C.CURRICULUM_DIR, f"{args.prefix}_bd{lead}.state")
        save_state(out, snap.state)
        saved.append((out, snap))

    for x_target in args.x_targets:
        snap = first_at_or_above(trace, x_target)
        if snap is None:
            print(f"skip x-target {x_target}: never reached")
            continue
        out = os.path.join(C.CURRICULUM_DIR, f"{args.prefix}_x{x_target}.state")
        save_state(out, snap.state)
        saved.append((out, snap))

    max_out = os.path.join(C.CURRICULUM_DIR, f"{args.prefix}_maxx.state")
    save_state(max_out, max_snap.state)
    saved.append((max_out, max_snap))

    print("\nsaved states:")
    for path, snap in saved:
        print(f"  {path}: step={snap.step} x={snap.x} y={snap.y}")


if __name__ == "__main__":
    main()
