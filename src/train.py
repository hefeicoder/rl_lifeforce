"""Train a PPO agent on Life Force Level 1.

Usage:
  python -m src.train                      # full run (config.TOTAL_TIMESTEPS)
  python -m src.train --timesteps 50000    # short run
  python -m src.train --smoke              # tiny end-to-end sanity run

Note: stable-retro allows only ONE emulator per process, so we use
SubprocVecEnv (one env per subprocess). DummyVecEnv would crash for N_ENVS > 1.
"""
import argparse
import os
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from . import config as C
from .env import make_thunk

VECNORM_PATH = os.path.join(C.CHECKPOINT_DIR, "vecnormalize.pkl")


class LifeForceStatsCallback(BaseCallback):
    """Log Life-Force-specific metrics to TensorBoard:

      reward/total, reward/score, reward/alive, reward/death, reward/clear
        -> per-episode reward broken into its components (rolling mean)
      lifeforce/stage_clears, lifeforce/clear_rate
        -> level-clear progress
    """

    def __init__(self, window=100):
        super().__init__()
        self._clears = 0
        self._episodes = 0
        self._comp = deque(maxlen=window)  # recent per-episode component dicts

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("stage_cleared"):
                self._clears += 1
            rc = info.get("reward_components")
            if rc is not None:             # set by LifeForceWrapper at episode end
                self._episodes += 1
                self._comp.append(rc)

        self.logger.record("lifeforce/stage_clears", self._clears)
        if self._episodes:
            self.logger.record("lifeforce/clear_rate", self._clears / self._episodes)
        if self._comp:
            n = len(self._comp)
            for k in self._comp[0]:              # score, alive, death, clear, powerup
                self.logger.record(f"reward/{k}", sum(c[k] for c in self._comp) / n)
            self.logger.record("reward/total",
                               sum(sum(c.values()) for c in self._comp) / n)
        return True


def build_vec_env(n_envs, load_norm=None):
    # Order: SubprocVecEnv -> VecMonitor (logs RAW episode returns) -> VecNormalize
    # (normalizes only what the algorithm trains on; raw reward_components in info
    # are untouched, so TensorBoard reward/* stays interpretable).
    base = VecMonitor(SubprocVecEnv([make_thunk(seed=i) for i in range(n_envs)]))
    if not C.NORM_REWARD:
        return base
    if load_norm and os.path.exists(load_norm):
        print(f"Loading reward-norm stats from {load_norm}")
        return VecNormalize.load(load_norm, base)
    return VecNormalize(base, norm_obs=False, norm_reward=True,
                        clip_reward=C.CLIP_REWARD, gamma=C.GAMMA)


def linear_schedule(initial, floor=0.0):
    """LR schedule: progress_remaining goes 1 -> 0, decaying `initial` down to
    `floor * initial` (floor=0.0 anneals all the way to zero)."""
    def f(progress_remaining):
        return (floor + (1.0 - floor) * progress_remaining) * initial
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=C.TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=C.N_ENVS)
    p.add_argument("--smoke", action="store_true", help="tiny sanity run")
    p.add_argument("--resume", default=None, help="path to a checkpoint .zip to continue from")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto'/'cpu'/'mps' (use 'mps' for Apple-GPU acceleration)")
    p.add_argument("--save-freq", type=int, default=C.CHECKPOINT_EVERY,
                   help="save a checkpoint every N total timesteps")
    args = p.parse_args()

    if args.smoke:
        args.timesteps, args.n_envs = 4096, 2

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    venv = build_vec_env(args.n_envs, load_norm=(VECNORM_PATH if args.resume else None))

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=venv, tensorboard_log=C.TB_LOG_DIR)
    else:
        lr = linear_schedule(C.LEARNING_RATE, C.LR_FLOOR) if C.LR_ANNEAL else C.LEARNING_RATE
        model = PPO(
            "CnnPolicy", venv,
            n_steps=C.N_STEPS, n_epochs=C.N_EPOCHS, batch_size=C.BATCH_SIZE,
            learning_rate=lr, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA,
            clip_range=C.CLIP_RANGE, ent_coef=C.ENT_COEF, target_kl=C.TARGET_KL,
            tensorboard_log=C.TB_LOG_DIR, verbose=1, device=args.device,
        )

    # CheckpointCallback's save_freq counts per-env steps, so divide the desired
    # total-timestep interval by the number of envs.
    ckpt = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=C.CHECKPOINT_DIR, name_prefix="lifeforce_ppo",
    )
    model.learn(total_timesteps=args.timesteps,
                callback=[ckpt, LifeForceStatsCallback()],
                progress_bar=True)

    final = os.path.join(C.CHECKPOINT_DIR, "lifeforce_ppo_final.zip")
    model.save(final)
    if isinstance(venv, VecNormalize):
        venv.save(VECNORM_PATH)          # reward-norm stats (for resume); play.py doesn't need it
    print(f"Saved final model -> {final}")
    venv.close()


if __name__ == "__main__":
    main()
