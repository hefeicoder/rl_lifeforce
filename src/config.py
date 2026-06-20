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
# Reduced discrete action set for a shmup. In Life Force firing has no cost, so
# shooting is weakly dominant (never worse than not shooting) -> we HARDWIRE fire
# (B) into every action and let the agent choose only movement. This removes a
# degree of freedom that should never be used, speeding learning.
# NES buttons available: B, SELECT, START, UP, DOWN, LEFT, RIGHT, A.
ACTIONS = [
    ["B"],                    # 0 fire, hold position
    ["UP", "B"],             # 1
    ["DOWN", "B"],           # 2
    ["LEFT", "B"],           # 3
    ["RIGHT", "B"],          # 4
    ["UP", "LEFT", "B"],     # 5
    ["UP", "RIGHT", "B"],    # 6
    ["DOWN", "LEFT", "B"],   # 7
    ["DOWN", "RIGHT", "B"],  # 8
    ["A", "B"],               # 9 activate power-up (gauge) while firing
]

# --- Reward shaping ----------------------------------------------------------
# Base reward (score-delta x10) comes from the bundled scenario.json. These add
# survival + level-clear shaping on top. Priorities: 1) stay alive  2) score
#  3) pass the level.
#
# Design note: survival is made #1 NOT by a big alive bonus (that just pays the
# agent to idle -> "camping") but by END_ON_LIFE_LOSS — dying forfeits all
# remaining reward, so staying alive is essential, and the only way to cash in on
# being alive is to SCORE. Score is therefore the main positive signal, which
# keeps play active and fun to watch. The alive bonus is just a small dense nudge.
REWARD_SCORE_SCALE = 1.0    # multiplier on base score reward (raise to weight scoring more)
REWARD_ALIVE = 0.02         # small dense survival nudge (NOT the main survival driver)
REWARD_DEATH = 5.0          # penalty subtracted when a life is lost
REWARD_CLEAR = 100.0        # bonus for clearing the stage (the goal)
END_ON_LIFE_LOSS = True     # the real "survival is #1" lever: death ends the episode
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
