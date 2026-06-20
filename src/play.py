"""Watch / evaluate a trained Life Force agent.

Records episodes to an MP4 (the live macOS window has a pyglet teardown bug, so
we render to rgb_array and write video instead) and reports per-episode score,
survival length, and whether Level 1 was cleared.

Usage:
  python -m src.play --model checkpoints/lifeforce_ppo_final.zip --episodes 3
"""
import argparse
import os

import imageio
import numpy as np
from stable_baselines3 import PPO

from . import config as C
from .env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to a trained .zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out", default=os.path.join(C.VIDEO_DIR, "play.mp4"))
    args = p.parse_args()

    os.makedirs(C.VIDEO_DIR, exist_ok=True)
    # render_mode="rgb_array" so we can grab full-res frames for the video.
    env = make_env(render_mode="rgb_array")
    model = PPO.load(args.model)

    frames, cleared_count = [], 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward, steps = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, info = env.step(int(action))
            frames.append(env.unwrapped.render())
            ep_reward += reward
            steps += 1
            done = term or trunc
        cleared = info.get("stage_cleared", False)
        cleared_count += int(cleared)
        print(f"ep {ep}: score={info.get('score')} steps={steps} "
              f"reward={ep_reward:.1f} max_x={info.get('x_pos')} "
              f"{'CLEARED LEVEL 1' if cleared else 'did not clear'}")

    print(f"\ncleared {cleared_count}/{args.episodes} episodes")
    imageio.mimsave(args.out, [np.asarray(f) for f in frames], fps=30)
    print(f"video -> {args.out}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
