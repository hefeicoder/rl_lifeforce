# RL Life Force (Salamander) — NES

Reinforcement learning on **Life Force** (the NES release of Konami's
*Salamander*), using [stable-retro](https://github.com/Farama-Foundation/stable-retro)
+ [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO.

**Goal:** train an agent to clear **Level 1**, with the project structured so
later levels are a config change rather than a rewrite.

Why this project exists: most RL game tutorials use turnkey packages (like
`gym-super-mario-bros`) that bundle the ROM, the action set, and the reward
signal. This one tackles the part those skip — doing RL on a game through a
**generic** emulator framework: building stable-retro from source, bringing your
own ROM, and extending the game integration (finding RAM addresses) yourself.

---

## Status

- ✅ **Feasibility proven.** stable-retro builds natively on Apple Silicon and
  runs Life Force Level 1 end-to-end. See [`docs/macos_arm64_build.md`](docs/macos_arm64_build.md).
- ✅ **RAM map** — confirmed lives/score/X/Y + auto-scroll clock; stage-clear
  detector narrowed to two suspects (`0x23`/`0x40`). See [`docs/ram_map.md`](docs/ram_map.md).
- ✅ **Training pipeline** — env factory (reduced action set + reward shaping +
  auto Stage-2 capture), PPO trainer, and video player; verified end-to-end.
- ⬜ Confirm the stage-clear detector from the first captured Stage-2 transition.
- ⬜ Train to clear Level 1.

## Train

Prerequisites: complete **Quickstart** (venv + `requirements.txt` +
`setup_stable_retro.sh`) and **import your ROM**. Then activate the venv:

```bash
source .venv/bin/activate
```

```bash
python -m src.train                       # full run (CPU)
python -m src.train --smoke               # tiny end-to-end sanity check
python -m src.play --model checkpoints/lifeforce_ppo_final.zip --episodes 3
```

**Device:** train on **CPU** (the default). This workload is *env-bound* —
stepping the NES emulators dominates, and NatureCNN is too small for a GPU to
help. Benchmarked on an M-series Mac, `--device mps` was ~25% *slower* (CPU↔GPU
transfer overhead with no compute win). The real speed lever is `N_ENVS` up to
your physical core count, not the GPU.

Reward reflects the objective (see `src/config.py`): **stay alive** (per-step
bonus + death penalty + one life per episode), **score** (base reward), and
**pass the level** (bonus on the Stage-2 transition, which also auto-captures
the Stage-2 RAM to finish confirming the clear detector).

## Quickstart

### 1. Install (Apple Silicon / macOS)

There is **no working prebuilt stable-retro wheel on Apple Silicon** — the
published one is a mislabeled x86_64 binary. Our script builds it natively from
source (NES core only). Full explanation: [`docs/macos_arm64_build.md`](docs/macos_arm64_build.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/setup_stable_retro.sh        # builds stable-retro from source
```

(On Linux, `pip install stable-retro` works directly.)

### 2. Bring your own ROM

This repo does **not** include the ROM — Life Force is copyrighted. Supply your
own legally-owned dump and import it:

```bash
python -m retro.import /path/to/your/roms/
```

stable-retro identifies the ROM by the SHA-1 of its *headerless* data
(`351edb1fdf4bce3bfc56d1eecccfdc6a21bb14f4`). Note this differs from
`shasum` of the `.nes` file, which includes the 16-byte iNES header.

### 3. Verify

```bash
python -c "import stable_retro as retro; env = retro.make('LifeForce-Nes-v0', state='1Player.Level1'); print(env.reset()[0].shape); env.close()"
```

## The interesting part: build & integration notes

We hit (and documented) three non-obvious blockers getting stable-retro running
on a current Apple-Silicon Mac — a mislabeled wheel, a removed Homebrew formula,
and modern clang refusing to compile the old Genesis/PCE cores. Full writeup:
[`docs/macos_arm64_build.md`](docs/macos_arm64_build.md).

## Licensing

- This project's code: MIT (see `LICENSE`, to be added).
- stable-retro: MIT. The NES core it builds (`fceumm`): **GPLv2** — which is why
  we ship a build *script*, not a prebuilt binary.
- ROMs: not included, not redistributable. Bring your own.
