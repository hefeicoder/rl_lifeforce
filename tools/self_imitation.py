"""Behaviour-clone agent-found demos into the policy, then hand back to PPO.

Loads a PPO checkpoint + Go-Explore demo(s) (obs/action pairs from
tools/explore_frontier.py) and supervised-trains the policy's action distribution to
reproduce the demo actions: negative log-likelihood over BOTH MultiDiscrete heads
(movement + activate). Small LR + few epochs so it SEEDS the discovered maneuver
without destroying the rest of the policy's learned behaviour. Save the result, then
resume normal PPO so the survival reward robustifies it (self-imitation loop).

Usage:
  python -m tools.self_imitation --model <ckpt> --demos demos/l3_bridge.npz \
    --out checkpoints/l3-bc/lifeforce_ppo_bc.zip
  # then:
  python -m src.train --resume checkpoints/l3-bc/lifeforce_ppo_bc.zip --run-name l3-bc-ppo
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
    print(f"next: python -m src.train --resume {args.out} --run-name <name> --ent-coef 0.05")


if __name__ == "__main__":
    main()
