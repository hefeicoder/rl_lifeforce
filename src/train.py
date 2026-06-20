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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from . import config as C
from .env import make_thunk


class LifeForceStatsCallback(BaseCallback):
    """Log Life-Force-specific metrics to TensorBoard: how far the agent gets
    (max x), and how often it clears the stage."""

    def __init__(self):
        super().__init__()
        self._clears = 0
        self._episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("stage_cleared"):
                self._clears += 1
            if "episode" in info:          # set by VecMonitor at episode end
                self._episodes += 1
        self.logger.record("lifeforce/stage_clears", self._clears)
        if self._episodes:
            self.logger.record("lifeforce/clear_rate", self._clears / self._episodes)
        return True


def build_vec_env(n_envs):
    return VecMonitor(SubprocVecEnv([make_thunk(seed=i) for i in range(n_envs)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=C.TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=C.N_ENVS)
    p.add_argument("--smoke", action="store_true", help="tiny sanity run")
    p.add_argument("--resume", default=None, help="path to a checkpoint .zip to continue from")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto'/'cpu'/'mps' (use 'mps' for Apple-GPU acceleration)")
    args = p.parse_args()

    if args.smoke:
        args.timesteps, args.n_envs = 4096, 2

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    venv = build_vec_env(args.n_envs)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=venv, tensorboard_log=C.TB_LOG_DIR)
    else:
        model = PPO(
            "CnnPolicy", venv,
            n_steps=C.N_STEPS, n_epochs=C.N_EPOCHS, batch_size=C.BATCH_SIZE,
            learning_rate=C.LEARNING_RATE, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA,
            clip_range=C.CLIP_RANGE, ent_coef=C.ENT_COEF,
            tensorboard_log=C.TB_LOG_DIR, verbose=1, device=args.device,
        )

    ckpt = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=C.CHECKPOINT_DIR, name_prefix="lifeforce_ppo",
    )
    model.learn(total_timesteps=args.timesteps,
                callback=[ckpt, LifeForceStatsCallback()],
                progress_bar=True)

    final = os.path.join(C.CHECKPOINT_DIR, "lifeforce_ppo_final.zip")
    model.save(final)
    print(f"Saved final model -> {final}")
    venv.close()


if __name__ == "__main__":
    main()
