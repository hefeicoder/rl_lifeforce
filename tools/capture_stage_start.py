"""Capture the emulator state immediately after a verified stage transition.

The current use is to turn the released deterministic Level-1 clear into a
canonical Level-2 reset artifact. The state remains a local, gitignored runtime
artifact because emulator save states contain ROM-derived data.

Example:
  python -m tools.capture_stage_start \
    --model checkpoints/l3-level1-clear/lifeforce_ppo_bc5_lr1e4.zip \
    --out states/l2_start.state --ram-out ram_dumps/l2_start_ram.npz
"""
import argparse
import gzip
import hashlib
import os

import numpy as np
from stable_baselines3 import PPO

from src import config as C
from src.env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="deterministic stage-clearing checkpoint")
    p.add_argument("--out", default=os.path.join(C.CURRICULUM_DIR, "l2_start.state"),
                   help="output gzipped emulator state")
    p.add_argument("--ram-out", default=os.path.join(C.RAM_DUMP_DIR, "l2_start_ram.npz"),
                   dest="ram_out", help="transition RAM/metadata capture")
    p.add_argument("--expected-vertical", type=int, default=1, dest="expected_vertical",
                   help="required ADDR_STAGE_VERTICAL value after transition")
    p.add_argument("--active-steps", type=int, default=2, dest="active_steps",
                   help="consecutive stable player-active steps before capture")
    p.add_argument("--force", action="store_true", help="replace existing output files")
    args = p.parse_args()

    existing = [path for path in (args.out, args.ram_out) if os.path.exists(path)]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite existing output(s): {existing}; pass --force")

    env = make_env(render_mode=None, curriculum=False)
    model = PPO.load(args.model, device="cpu")
    obs, info = env.reset(seed=0)
    steps = 0
    transition_steps = None
    captured_state = None
    captured_ram = None
    stable_active = 0
    prev_lives = int(env.unwrapped.get_ram()[C.ADDR_LIVES])

    while steps < C.MAX_EPISODE_STEPS:
        if transition_steps is None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Input is ignored during most of the fly-in. HOLD avoids baking an
            # old horizontal policy's movement into the new vertical-stage start.
            action = np.array([0, 0])
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        ram = env.unwrapped.get_ram()
        if info.get("stage_cleared") and transition_steps is None:
            transition_steps = steps

        if transition_steps is not None:
            lives = int(ram[C.ADDR_LIVES])
            active = (
                int(ram[C.ADDR_CTRL]) == 3
                and C.X_POS_MIN <= int(ram[C.ADDR_X_POS]) <= C.X_POS_MAX
                and 0 < int(ram[C.ADDR_Y_POS]) < 240
                and lives == prev_lives
            )
            stable_active = stable_active + 1 if active else 0
            if stable_active >= args.active_steps:
                captured_state = env.unwrapped.em.get_state()
                captured_ram = ram.copy()
                break

        # Stage-transition and fly-in bookkeeping may set terminated while RAM
        # is still initializing. Ignore it after the verified transition; before
        # the transition, any termination is a real failure.
        if transition_steps is None and (terminated or truncated):
            reason = ("life loss" if info.get("life_lost") else
                      "time limit" if truncated else "termination")
            raise SystemExit(
                f"model reached {reason} after {steps} steps before a stage transition"
            )
        prev_lives = int(ram[C.ADDR_LIVES])

    if captured_state is None:
        raise SystemExit(f"no stage transition within {C.MAX_EPISODE_STEPS} steps")

    vertical = int(captured_ram[C.ADDR_STAGE_VERTICAL])
    if vertical != args.expected_vertical:
        raise SystemExit(
            f"transition vertical flag is {vertical}, expected {args.expected_vertical}; not saving"
        )

    state_sha256 = hashlib.sha256(captured_state).hexdigest()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wb") as fh:
        fh.write(captured_state)

    os.makedirs(os.path.dirname(args.ram_out) or ".", exist_ok=True)
    np.savez_compressed(
        args.ram_out,
        ram=captured_ram,
        steps=np.int64(steps),
        transition_steps=np.int64(transition_steps),
        score=np.int64(info.get("score", 0)),
        stage_num=np.int64(captured_ram[C.ADDR_STAGE_NUM]),
        stage_vertical=np.int64(vertical),
        player_control=np.int64(captured_ram[C.ADDR_CTRL]),
        state_sha256=np.asarray(state_sha256),
        model=np.asarray(args.model),
    )

    print(f"transition detected at step {transition_steps}; "
          f"settled active start captured at step {steps} / score={info.get('score')}")
    print(f"RAM: stage={int(captured_ram[C.ADDR_STAGE_NUM])} "
          f"vertical={vertical} control={int(captured_ram[C.ADDR_CTRL])} "
          f"x={int(captured_ram[C.ADDR_X_POS])} y={int(captured_ram[C.ADDR_Y_POS])}")
    print(f"state -> {args.out}")
    print(f"raw-state SHA-256: {state_sha256}")
    print(f"RAM metadata -> {args.ram_out}")
    print("next:")
    print(f"  python -m src.play --model {args.model} --initial-state {args.out} "
          "--deterministic --episodes 3")
    env.close()


if __name__ == "__main__":
    main()
