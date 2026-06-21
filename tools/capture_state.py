"""Capture a save-state at a hard section, for the curriculum.

Runs a trained agent from the level start and saves the emulator state from
shortly BEFORE it dies — i.e. the spot it's stuck at. The agent's own failure
point defines the wall, so you don't have to know where it is. Saves to
CURRICULUM_DIR/<name>.state, after which `python -m src.train --resume ...`
automatically mixes it into training (see CurriculumStart in env.py).

Usage:
  python -m tools.capture_state --model checkpoints/lifeforce_ppo_<N>_steps.zip --name l1_gauntlet
  python -m tools.capture_state --model ... --name l1_gauntlet --episodes 8 --before-death 30
"""
import argparse
import gzip
import os
from collections import deque

from stable_baselines3 import PPO

from src import config as C
from src.env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint to drive the capture")
    p.add_argument("--name", required=True, help="output name -> CURRICULUM_DIR/<name>.state")
    p.add_argument("--episodes", type=int, default=8, help="episodes to try; keep the furthest")
    p.add_argument("--before-death", type=int, default=30, dest="before_death",
                   help="save the state this many agent-steps before death (the wall lead-in)")
    p.add_argument("--sample", action="store_true",
                   help="sample actions instead of deterministic (default deterministic)")
    args = p.parse_args()

    env = make_env(render_mode=None, curriculum=False)  # always start from the real level start
    em = env.unwrapped.em
    model = PPO.load(args.model)

    best_steps, best_state = -1, None
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        ring = deque(maxlen=args.before_death + 1)  # states leading up to "now"
        steps, done = 0, False
        last_info = {}
        while not done:
            ring.append(em.get_state())
            action, _ = model.predict(obs, deterministic=not args.sample)
            obs, _, term, trunc, last_info = env.step(action)
            steps += 1
            done = term or trunc
        if steps > best_steps:               # keep the furthest-reaching run (the consistent wall)
            best_steps, best_state = steps, ring[0]
        print(f"ep {ep}: survived {steps} steps  (score={last_info.get('score')})")

    os.makedirs(C.CURRICULUM_DIR, exist_ok=True)
    out = os.path.join(C.CURRICULUM_DIR, f"{args.name}.state")
    with gzip.open(out, "wb") as fh:
        fh.write(best_state)
    print(f"\nCaptured ~{args.before_death} steps before death of the furthest run "
          f"({best_steps} steps) -> {out}")
    print("Replay this state to verify it's a fair approach (not a cornered frame):")
    print(f"  python -m src.play --model {args.model} --from-state {out}")
    print("Then resume training; it will mix this state in automatically:")
    print(f"  python -m src.train --resume {args.model}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
