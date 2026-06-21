"""Throughput benchmark for the training env stack.

Measures agent-steps/sec (and emulated fps = ×FRAME_SKIP) for the real training
vec-env stack (SubprocVecEnv -> VecMonitor -> VecNormalize) across N_ENVS, and
isolates the policy-predict cost vs raw env stepping. Use it to find whether we're
env-bound (more envs won't help) and where the per-step time actually goes.

  python -m tools.bench                       # env throughput sweep
  python -m tools.bench --model <ckpt.zip>    # also measure predict overhead
"""
import argparse
import time

import numpy as np

from src import config as C
from src.train import build_vec_env


def measure(venv, n_envs, steps, rng):
    venv.reset()
    # warmup (spawn/JIT/first-frame costs out of the timing)
    for _ in range(30):
        venv.step(np.array([[rng.integers(9), rng.integers(2)] for _ in range(n_envs)]))
    t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(np.array([[rng.integers(9), rng.integers(2)] for _ in range(n_envs)]))
    dt = time.perf_counter() - t0
    agent_sps = steps * n_envs / dt          # agent steps/sec across all envs
    return agent_sps, agent_sps * C.FRAME_SKIP  # emulated fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, nargs="+", default=[8, 10, 12, 16])
    p.add_argument("--steps", type=int, default=1200, help="timed agent-steps per env per config")
    p.add_argument("--model", default=None, help="checkpoint to measure predict overhead")
    args = p.parse_args()
    rng = np.random.default_rng(0)

    print(f"FRAME_SKIP={C.FRAME_SKIP}  timed steps/env={args.steps}\n")
    print(f"{'N_ENVS':>6} {'agent_sps':>10} {'emul_fps':>9} {'per_env_fps':>11}")
    best = None
    for n in args.envs:
        venv = build_vec_env(n)
        asps, fps = measure(venv, n, args.steps, rng)
        venv.close()
        print(f"{n:>6} {asps:>10.0f} {fps:>9.0f} {fps / n:>11.0f}")
        if best is None or asps > best[1]:
            best = (n, asps)
        time.sleep(0.5)
    print(f"\nbest: N_ENVS={best[0]} at {best[1]:.0f} agent-sps")

    if args.model:
        from stable_baselines3 import PPO
        n = best[0]
        venv = build_vec_env(n)
        obs = venv.reset()
        model = PPO.load(args.model, device="cpu")
        # predict-only
        t0 = time.perf_counter()
        for _ in range(300):
            model.predict(obs, deterministic=False)
        t_pred = (time.perf_counter() - t0) / 300
        # step-only (random)
        t0 = time.perf_counter()
        for _ in range(300):
            venv.step(np.array([[rng.integers(9), rng.integers(2)] for _ in range(n)]))
        t_step = (time.perf_counter() - t0) / 300
        venv.close()
        print(f"\npredict overhead @ N_ENVS={n}:")
        print(f"  env.step():     {t_step*1000:7.2f} ms/agent-step")
        print(f"  model.predict():{t_pred*1000:7.2f} ms/agent-step")
        print(f"  predict is {t_pred/(t_pred+t_step)*100:.0f}% of the rollout step")


if __name__ == "__main__":
    main()
