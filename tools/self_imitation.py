"""Behaviour-clone agent-found golden trajectories into the policy.

Loads a PPO checkpoint plus demo(s) containing observation/action pairs and
supervised-trains the policy's action distribution to reproduce them: negative
log-likelihood over both MultiDiscrete heads (movement + activate). The current
standard is a complete canonical reset-to-continuation trajectory recorded by
``segment_search --record-continuation``. Use a small LR/few epochs, then evaluate
the clone from the real reset before considering optional PPO robustness training.

Usage:
  python -m tools.self_imitation --model <ckpt> --demos demos/next_frontier.npz \
    --out checkpoints/next/lifeforce_ppo_bc.zip --epochs 5 --lr 1e-4
  python -m src.play --model checkpoints/next/lifeforce_ppo_bc.zip \
    --deterministic --episodes 3
"""
import argparse
import glob
import os

import numpy as np
import torch
from stable_baselines3 import PPO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--demos", nargs="+", required=True, help="demo .npz file(s) / globs (obs, actions)")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4, help="keep small: too large wrecks the rest of the policy")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    paths = []
    for pat in args.demos:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit(f"no demo files matched {args.demos}")
    obs = np.concatenate([np.load(x)["obs"] for x in paths]).astype(np.uint8)   # (N, C, H, W)
    acts = np.concatenate([np.load(x)["actions"] for x in paths])               # (N, n_heads)
    print(f"loaded {len(obs)} demo steps from {len(paths)} file(s): {paths}")

    model = PPO.load(args.model, device=args.device)
    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    N = len(obs)
    for epoch in range(args.epochs):
        idx = np.random.permutation(N)
        total, nb = 0.0, 0
        for s in range(0, N, args.batch_size):
            b = idx[s:s + args.batch_size]
            obs_t, _ = policy.obs_to_tensor(obs[b])
            act_t = torch.as_tensor(acts[b], device=policy.device).long()
            dist = policy.get_distribution(obs_t)
            loss = -dist.log_prob(act_t).mean()        # NLL over both MultiDiscrete heads
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"epoch {epoch + 1}/{args.epochs}  BC loss {total / nb:.4f}")

    policy.set_training_mode(False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"saved BC'd model -> {args.out}")
    print(f"next: python -m src.play --model {args.out} --deterministic --episodes 3 "
          "[--initial-state <stage-start.state>]")


if __name__ == "__main__":
    main()
