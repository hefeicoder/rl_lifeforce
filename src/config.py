"""Central config for the Life Force RL project.

One place for the game/integration constants, the RAM addresses we found (see
docs/ram_map.md), the action set, the reward shaping weights, and PPO
hyperparameters. Tune training behavior here, not scattered across modules.
"""

# --- Game / integration ------------------------------------------------------
GAME = "LifeForce-Nes-v0"
STATE = "1Player.Level1"

# RAM addresses (decimal). Confirmed via tools/ram_hunt.py + Data Crystal.
ADDR_LIVES = 0x34          # P1 lives (also exposed in info by bundled data.json)
ADDR_X_POS = 0x350         # P1 X position (15..232)
ADDR_Y_POS = 0x32F         # P1 Y position (24..197)
ADDR_STAGE_NUM = 0x23      # "Demo Stage Num?" — stage-transition suspect
ADDR_STAGE_VERTICAL = 0x40  # "Is Stage Vertical?" — flips 0->1 on the vertical Stage 2

# --- Action set --------------------------------------------------------------
# Reduced discrete action set for a shmup, given as NES button-name combos.
# We almost always want to be firing (B), so every move is paired with fire.
# NES buttons available: B, SELECT, START, UP, DOWN, LEFT, RIGHT, A.
ACTIONS = [
    [],                       # 0 no-op
    ["B"],                    # 1 fire, hold position
    ["UP", "B"],             # 2
    ["DOWN", "B"],           # 3
    ["LEFT", "B"],           # 4
    ["RIGHT", "B"],          # 5
    ["UP", "LEFT", "B"],     # 6
    ["UP", "RIGHT", "B"],    # 7
    ["DOWN", "LEFT", "B"],   # 8
    ["DOWN", "RIGHT", "B"],  # 9
    ["A"],                    # 10 activate power-up (Gradius-style gauge)
]

# --- Reward shaping ----------------------------------------------------------
# Base reward (score-delta x10) comes from the bundled scenario.json. These add
# survival + level-clear shaping on top. See README for the priorities:
#   1) stay alive  2) score  3) pass the level.
REWARD_ALIVE = 0.1          # per agent-step bonus for staying alive
REWARD_DEATH = 5.0          # penalty subtracted when a life is lost
REWARD_CLEAR = 50.0         # bonus when the Stage-1 -> Stage-2 transition is seen
END_ON_LIFE_LOSS = True     # terminate the episode on the first death (1 life/episode)
MAX_EPISODE_STEPS = 2000    # agent-step time limit (post frame-skip)

# --- Preprocessing -----------------------------------------------------------
FRAME_SKIP = 4
FRAME_SIZE = 84             # bullets are small; bump this if the agent can't "see" threats
FRAME_STACK = 4

# --- PPO / training ----------------------------------------------------------
N_ENVS = 8
N_STEPS = 512
N_EPOCHS = 4
BATCH_SIZE = 1024
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.1
ENT_COEF = 0.01
TOTAL_TIMESTEPS = 5_000_000

# --- Paths -------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
TB_LOG_DIR = "tb_logs"
VIDEO_DIR = "videos"
RAM_DUMP_DIR = "ram_dumps"
