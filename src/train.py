"""Train a PPO agent on Life Force Level 1.

Usage:
  python -m src.train                      # full run (config.TOTAL_TIMESTEPS)
  python -m src.train --timesteps 50000    # short run
  python -m src.train --smoke              # tiny end-to-end sanity run

Note: stable-retro allows only ONE emulator per process, so we use
SubprocVecEnv (one env per subprocess). DummyVecEnv would crash for N_ENVS > 1.
"""
import argparse
import datetime
import os
import re
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from . import config as C
from .env import make_thunk

VECNORM_NAME = "vecnormalize.pkl"   # canonical name (final save)


def vecnorm_for(ckpt):
    """Find the reward-norm stats saved beside a checkpoint, for --resume.

    CheckpointCallback(save_vecnormalize=True) writes per-checkpoint stats named
    '<prefix>_vecnormalize_<N>_steps.pkl' next to each '<prefix>_<N>_steps.zip', so
    any mid-run checkpoint is independently resumable. The final save writes the
    canonical 'vecnormalize.pkl'. Prefer the exact per-checkpoint file, else fall
    back to the canonical name (also covers '..._final.zip').
    """
    d = os.path.dirname(ckpt) or "."
    m = re.match(r"(.+?)_(\d+)_steps\.zip$", os.path.basename(ckpt))
    if m:
        cand = os.path.join(d, f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl")
        if os.path.exists(cand):
            return cand
    return os.path.join(d, VECNORM_NAME)


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
        self._comp = deque(maxlen=window)    # recent per-episode reward-component dicts
        self._best_score = 0                 # all-time max ABSOLUTE in-game score (progress marker)
        self._recent = deque(maxlen=window)  # recent end-of-episode absolute scores

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("stage_cleared"):
                self._clears += 1
            rc = info.get("reward_components")
            if rc is not None:             # set by LifeForceWrapper at episode end
                self._episodes += 1
                self._comp.append(rc)
                sc = int(info.get("score", 0))   # absolute score — the save state preserves it,
                self._recent.append(sc)          # so "best ever" rising past the wall (~2620)
                self._best_score = max(self._best_score, sc)  # = a breakthrough, regardless of start

        self.logger.record("lifeforce/stage_clears", self._clears)
        if self._episodes:
            self.logger.record("lifeforce/clear_rate", self._clears / self._episodes)
        self.logger.record("lifeforce/best_score", self._best_score)        # all-time
        if self._recent:
            self.logger.record("lifeforce/recent_best_score", max(self._recent))  # current capability
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


def resolve_device(device):
    """Resolve --device. SB3's own 'auto' only ever picks CUDA-or-CPU (never MPS),
    so on Apple Silicon 'auto' would silently train on CPU. We benchmarked the PPO
    learn phase (batch-1024 conv backprop) at ~85% of wall-clock and MPS ~2.5x
    faster end-to-end than CPU (and CPU thermally throttles on long runs), so we
    prefer MPS when available. Pass --device cpu to force CPU."""
    if device == "auto":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=C.TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=C.N_ENVS)
    p.add_argument("--smoke", action="store_true", help="tiny sanity run")
    p.add_argument("--resume", default=None, help="path to a checkpoint .zip to continue from")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto' (=mps on Apple Silicon, ~2.5x faster) / 'cpu' / 'mps'")
    p.add_argument("--save-freq", type=int, default=C.CHECKPOINT_EVERY,
                   help="save a checkpoint every N total timesteps")
    p.add_argument("--run-name", default=None, dest="run_name",
                   help="name for this run's output folder (default: timestamp)")
    p.add_argument("--ent-coef", type=float, default=None, dest="ent_coef",
                   help="override entropy coefficient (higher = more exploration; default config.ENT_COEF)")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="minibatch size (default config.BATCH_SIZE); larger = better GPU utilization")
    p.add_argument("--n-steps", type=int, default=None, dest="n_steps",
                   help="rollout length per env (default config.N_STEPS); fresh runs only")
    p.add_argument("--n-epochs", type=int, default=None, dest="n_epochs",
                   help="PPO passes per rollout (default config.N_EPOCHS)")
    args = p.parse_args()

    if args.smoke:
        args.timesteps, args.n_envs = 4096, 2

    # Each run gets its own folder so resumed runs don't overwrite each other's
    # checkpoints (SB3 resets the step counter on resume). The TensorBoard run
    # uses the same name, so curves and checkpoints line up.
    run_name = args.run_name or datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = os.path.join(C.CHECKPOINT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_name}  ->  checkpoints in {run_dir}/ , TensorBoard run '{run_name}'")

    # On resume, load that checkpoint's normalization stats from beside it.
    load_norm = vecnorm_for(args.resume) if args.resume else None
    venv = build_vec_env(args.n_envs, load_norm=load_norm)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    ent_coef = args.ent_coef if args.ent_coef is not None else C.ENT_COEF
    batch_size = args.batch_size or C.BATCH_SIZE
    n_steps = args.n_steps or C.N_STEPS
    n_epochs = args.n_epochs or C.N_EPOCHS
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=venv, tensorboard_log=C.TB_LOG_DIR, device=device)
        model.ent_coef = ent_coef          # allow raising exploration on resume
        model.batch_size, model.n_epochs = batch_size, n_epochs  # safe to change on resume
        if args.n_steps and args.n_steps != model.n_steps:
            print("warning: --n-steps only takes effect on a fresh run (it sizes the "
                  "rollout buffer at construction); ignoring on resume")
    else:
        lr = linear_schedule(C.LEARNING_RATE, C.LR_FLOOR) if C.LR_ANNEAL else C.LEARNING_RATE
        model = PPO(
            "CnnPolicy", venv,
            n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size,
            learning_rate=lr, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA,
            clip_range=C.CLIP_RANGE, ent_coef=ent_coef, target_kl=C.TARGET_KL,
            tensorboard_log=C.TB_LOG_DIR, verbose=1, device=device,
        )

    # CheckpointCallback's save_freq counts per-env steps, so divide the desired
    # total-timestep interval by the number of envs.
    # save_vecnormalize=True writes the reward-norm stats next to each checkpoint,
    # so any mid-run checkpoint is independently resumable (vecnorm_for finds them).
    ckpt = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=run_dir, name_prefix="lifeforce_ppo",
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=args.timesteps,
                callback=[ckpt, LifeForceStatsCallback()],
                progress_bar=True, tb_log_name=run_name)

    final = os.path.join(run_dir, "lifeforce_ppo_final.zip")
    model.save(final)
    if isinstance(venv, VecNormalize):
        venv.save(os.path.join(run_dir, VECNORM_NAME))  # reward-norm stats (for resume)
    print(f"Saved final model -> {final}")
    venv.close()


if __name__ == "__main__":
    main()
