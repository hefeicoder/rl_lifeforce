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

# Power-up / weapon state (Data Crystal, verified via tools/ram_hunt). The agent
# can read these to learn the Gradius-style meter: collect capsules (0x78 cursor
# advances), then spend to gain upgrades. Caps are inherent (e.g. options <= 2).
ADDR_POWERBAR = 0x78       # power-bar cursor (1=speed,2=missile,3=ripple,4=laser,5=option,6=force field)
SPEED_SLOT = 1             # cursor value where activating gives Speed (verified)
ADDR_SPEED = 0x80          # speed level (up to 10)
ADDR_SHIELD = 0x82         # Force Field / shield strength (starts at 5 hits when activated)
ADDR_MISSILE = 0x86        # Missile equipped
ADDR_OPTIONS = 0x8A        # Options / Multiples (up to 2)
ADDR_WEAPON = 0x76         # 0=Normal, 1=Ripple, 2=Laser
ADDR_CTRL = 0x70           # player control state (3=active, 4/5=dying, 1/2=flying in)

# --- Action set --------------------------------------------------------------
# Action space: MultiDiscrete([len(MOVES), 2]) — two INDEPENDENT decisions:
#   head 1: movement (fire B is hardwired on, since shooting is weakly dominant)
#   head 2: press the power-up button (A) or not
# This lets the agent activate a power-up WHILE moving (faithful to the game and
# removes the survival-vs-activate conflict). Factoring the two decisions is more
# sample-efficient than a flat Discrete(18): the "when to activate" policy is
# learned once and generalizes across all movements.
# NES buttons available: B, SELECT, START, UP, DOWN, LEFT, RIGHT, A.
MOVES = [
    ["B"],                    # 0 hold position (fire)
    ["UP", "B"],             # 1
    ["DOWN", "B"],           # 2
    ["LEFT", "B"],           # 3
    ["RIGHT", "B"],          # 4
    ["UP", "LEFT", "B"],     # 5
    ["UP", "RIGHT", "B"],    # 6
    ["DOWN", "LEFT", "B"],   # 7
    ["DOWN", "RIGHT", "B"],  # 8
]
ACTIVATE_BUTTON = "A"        # pressed when the activate head outputs 1

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

# Power-up shaping (one-time bonuses on STATE INCREASES, so upgrade caps are
# self-enforcing — a maxed upgrade can't increase, so it earns nothing and the
# agent learns not to waste capsules on it). Teaches: eat capsules, accumulate,
# spend well. Priority for scoring: Missile > Option > Force Field.
# Loadout priority: Missile > Option > Force Field. SPEED is HARD-CAPPED at
# MAX_SPEED: the Discretizer (env.py) refuses to activate the power-up on the
# speed slot once speed >= MAX_SPEED, so the agent physically can't over-speed.
# Why a hard cap and not a penalty: speed is net-positive early (dodge -> survive
# -> score), and the place it hurts (gauntlet overshoot) is ~600 steps away,
# γ-discounted to nothing — so the agent kept over-speeding even at a -5/level
# penalty. The REWARD_SPEED/REWARD_OVERSPEED weights below are now a backstop;
# the mask is the real enforcer.
REWARD_CAPSULE = 0.5        # ate a capsule (cursor advanced) — currency for upgrades
REWARD_MISSILE = 4.0        # acquired Missile (top priority)
REWARD_OPTION = 3.0         # acquired an Option (max 2)
REWARD_FORCEFIELD = 2.0     # gained Force Field / refilled shield
MAX_SPEED = 2              # speed levels up to here are "good"; beyond = over-speeding
REWARD_SPEED = 0.5         # bonus per speed level gained up to MAX_SPEED
REWARD_OVERSPEED = -5.0    # penalty per speed level gained ABOVE MAX_SPEED (much heavier)

# Forward-position reward (experimental, for the fork). Small per-step bonus scaled
# by how far FORWARD (high x) the ship is. Rationale: the fork's dead-end channel
# corners the ship to the far back (x~16) before death, so this pays LESS there — a
# dense, EARLY "bad" signal (vs the late, discounted death) — and rewards the
# front-rush survival line. NOTE: x_pos is SCREEN position, not level distance (the
# level auto-scrolls). CAUTION: conflicts with the gauntlet-#1 lesson (front edge =
# death there) — watch best_score for regression below 380. 0.0 disables.
REWARD_XPOS = 0.02

# --- Positional cap (action mask, like the speed cap) ------------------------
# Hugging the leading (front/right) edge leaves no time to react to terrain and
# enemies that scroll in from the front — which is exactly how the agent dies at
# the gauntlet. So we MASK the RIGHT button once the ship is at/forward of
# X_SAFE_FRONT: it physically cannot advance past the back zone (it can still
# hold position, retreat, and move vertically). A mask, not a positional penalty,
# for the same reason the speed cap is a mask — a penalty fights an arms race.
# x_pos ranges ~15 (far back/left) .. 232 (front/right edge).
X_POS_MIN = 15
X_POS_MAX = 232
# X_SAFE_FRONT: mask RIGHT at/forward of this x (keep the ship back). None = cap OFF
# (full positional freedom). Set to 100 (back ~40%) it SOLVED gauntlet #1 (front-hugging
# = no reaction time there). But it may FUNNEL the ship into the fork's dead-end channel
# by forbidding front positioning — so we're testing cap-off + higher exploration to see
# if the agent can find a passing line the cap was hiding. Fallback if gauntlet #1
# regresses: score-gate the cap (on for score < ~250, off at the fork).
X_SAFE_FRONT = None

# --- Preprocessing -----------------------------------------------------------
FRAME_SKIP = 4
FRAME_SIZE = 84             # bullets are small; bump this if the agent can't "see" threats
FRAME_STACK = 4

# --- PPO / training ----------------------------------------------------------
# N_ENVS is the #1 throughput lever on MPS (see docs/devlog.md): the bottleneck is
# per-step policy inference (CPU<->GPU transfer), and more envs amortize it over a
# bigger batch per call. Benchmarked on M1 Max: 8->629, 16->943, 32->1378 fps.
# Cost: rollout-buffer memory grows with N_ENVS, and very large counts can slightly
# dent sample efficiency. 16 is a safe 1.5x; bump via --n-envs 24/32 for raw speed.
N_ENVS = 16
N_STEPS = 512
N_EPOCHS = 4
BATCH_SIZE = 1024
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.1
ENT_COEF = 0.01
TARGET_KL = 0.02             # early-stop an update if policy moves too far (anti-collapse)
NORM_REWARD = True           # VecNormalize returns to ~unit variance (stabilizes the critic
                             # against large/high-variance rewards -> damps reward oscillation)
CLIP_REWARD = 10.0           # clip the normalized reward to this range
LR_ANNEAL = True             # linearly decay learning rate over training (stability)
LR_FLOOR = 0.1               # anneal down to this fraction of initial LR (0.0 = all the way to 0).
                             # A small floor keeps the policy learning at the end of a run, which
                             # suits exploratory/unknown-horizon training; use 0.0 for a final run.
TOTAL_TIMESTEPS = 2_000_000  # a CEILING, not a commitment: runs plateau early, so watch
                             # best_score and Ctrl-C when flat (latest checkpoint is resumable).
                             # ~35 min at ~950 fps; use --timesteps 5M for a deliberate long run.
CHECKPOINT_EVERY = 100_000   # save a checkpoint every N total timesteps (--save-freq)

# --- Curriculum (save-state practice for hard sections) ----------------------
# Drop-in design: any *.state file in CURRICULUM_DIR becomes a possible start
# state. Capture one at a wall with tools/capture_state.py, then just resume —
# no code/config edits per wall. The level's real start is always kept too, so
# the agent still learns the whole stage and we can measure true clears.
CURRICULUM_DIR = "states"   # directory of *.state files (gitignored; regenerate via capture)
CURRICULUM_MIX = 0.5        # P(episode starts from a random curriculum state vs the level start)

# --- Paths -------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
TB_LOG_DIR = "tb_logs"
VIDEO_DIR = "videos"
RAM_DUMP_DIR = "ram_dumps"
