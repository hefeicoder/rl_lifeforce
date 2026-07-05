"""Life Force environment factory.

Builds a Gymnasium/SB3-ready env from stable-retro:

    retro env (MultiBinary(9), 224x240x3)
      -> Discretizer        (Discrete action set -> button presses)
      -> MaxAndSkip         (act every FRAME_SKIP frames)
      -> LifeForceWrapper   (RAM-based reward shaping, done, Stage-2 capture)
      -> Grayscale/Resize/FrameStack
      -> TimeLimit

stable-retro is natively Gymnasium-compatible, so unlike the Mario project no
shimmy/compat shims are needed.
"""
import glob
import gzip
import os

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipObservation,
    ResizeObservation,
)
import stable_retro as retro

from . import config as C


def _x_frac(x):
    """Normalized screen-x in [0,1]. Clamped: x_pos (RAM 0x350) can momentarily
    exceed the documented X_POS_MAX (e.g. 234>232) at scroll transitions, which
    would otherwise push the ratchet potential above 1."""
    return min(1.0, max(0.0, (x - C.X_POS_MIN) / (C.X_POS_MAX - C.X_POS_MIN)))


class CurriculumStart(gym.Wrapper):
    """On reset, start either from the level's default state or — with prob
    ``mix`` — a random saved state in ``curriculum_dir`` (so the agent can drill
    hard sections). The directory is re-scanned every reset, so new .state files
    (from tools/capture_state, or a future auto-curriculum) are picked up live
    without restarting training.
    """

    def __init__(self, env, glob_pattern, mix, rng_seed=0):
        super().__init__(env)
        self._pattern = glob_pattern   # e.g. "states/*.state" or "states/l3_wall.state"
        self._mix = mix
        self._rng = np.random.default_rng(rng_seed)
        self._default_state = None  # the level's real start, captured on first reset

    def reset(self, **kwargs):
        u = self.env.unwrapped
        if self._default_state is None:
            self._default_state = u.initial_state
        states = sorted(glob.glob(self._pattern)) if self._pattern else []
        if states and self._rng.random() < self._mix:
            path = states[self._rng.integers(len(states))]
            with gzip.open(path, "rb") as fh:
                u.initial_state = fh.read()
            # tag start source so metrics can split L3-wall episodes from full-level
            # starts (whose ~880-step survival otherwise pollutes best_steps).
            u._curriculum_start = True
            u._curriculum_label = os.path.splitext(os.path.basename(path))[0]
        else:
            u.initial_state = self._default_state
            u._curriculum_start = False
            u._curriculum_label = "level_start"
        return self.env.reset(**kwargs)


class Discretizer(gym.ActionWrapper):
    """Map a MultiDiscrete([n_moves, 2]) action to the NES button vector.

    Decision 1 = movement (fire B hardwired on); decision 2 = press the power-up
    button (A) or not. So the agent can activate a power-up while moving.
    """

    def __init__(self, env, moves, activate_button):
        super().__init__(env)
        buttons = env.unwrapped.buttons  # e.g. ['B', None, 'SELECT', ...]
        self._moves = []
        for combo in moves:
            arr = np.zeros(len(buttons), dtype=np.int8)
            for name in combo:
                arr[buttons.index(name)] = 1
            self._moves.append(arr)
        self._activate_idx = buttons.index(activate_button)
        self._right_idx = buttons.index("RIGHT")
        self.action_space = gym.spaces.MultiDiscrete([len(self._moves), 2])

    def action(self, act):
        move_idx, activate = int(act[0]), int(act[1])
        ram = self.env.unwrapped.get_ram()
        arr = self._moves[move_idx].copy()
        # Under AUTO_BUY the agent's activate head is IGNORED (vestigial, like the
        # hardwired B): random A presses were measured spending the banked meter at
        # whatever slot the cursor happened to sit on (e.g. Laser at 4).
        if activate and not C.AUTO_BUY and not self._would_overspeed(ram):
            arr[self._activate_idx] = 1
        # AUTO_BUY: the meter strategy is fixed (Missile once, then Options) and
        # timing an A press at the exact cursor stop is a needle-thin credit
        # assignment RL never solved (measured: a policy that banks 6 capsules and
        # never spends). Same design language as hardwiring B: no interesting
        # decision, so don't ask the agent to learn it.
        if C.AUTO_BUY:
            bar = int(ram[C.ADDR_POWERBAR])
            if ((bar == C.MISSILE_SLOT and int(ram[C.ADDR_MISSILE]) == 0)
                    or (bar == C.OPTION_SLOT and int(ram[C.ADDR_OPTIONS]) < 2)):
                arr[self._activate_idx] = 1
        if self._too_far_front(ram):
            arr[self._right_idx] = 0   # positional cap: no advancing past the back zone
        return arr

    def _would_overspeed(self, ram):
        # Hard cap: refuse to activate the power-up when the meter cursor is on the
        # SPEED slot and speed is already at MAX_SPEED. Too much speed makes the
        # ship overshoot in tight terrain, and a reward penalty alone can't stop
        # the agent (speed is net-positive early), so we prevent it outright.
        return int(ram[C.ADDR_POWERBAR]) == C.SPEED_SLOT and int(ram[C.ADDR_SPEED]) >= C.MAX_SPEED

    def _too_far_front(self, ram):
        # Hard cap on forward position: drop RIGHT once the ship is at/forward of
        # X_SAFE_FRONT, so it can't hug the leading edge (no reaction time to
        # terrain scrolling in — the gauntlet death). It can still retreat, hover,
        # and move vertically. A mask, not a penalty, for the same reason as the
        # speed cap. X_SAFE_FRONT=None disables the cap (full positional freedom).
        if C.X_SAFE_FRONT is None:
            return False
        return int(ram[C.ADDR_X_POS]) >= C.X_SAFE_FRONT


class LifeForceWrapper(gym.Wrapper):
    """RAM-based reward shaping + episode logic for Life Force.

    Reads the addresses found during the RAM hunt (see docs/ram_map.md) and
    encodes the project's objective: stay alive, score, pass the level. Also
    auto-captures the Stage-1 -> Stage-2 transition RAM the first time it is
    seen, which is how we finish confirming the stage-clear detector.
    """

    def __init__(self, env, reward_score_scale=None, reward_alive=None,
                 reward_death=None, reward_xpos=None, x_front_frac=None,
                 reward_xmax=None, reward_move_cost=None, reward_churn=None):
        super().__init__(env)
        self._captured = False
        self._reward_score_scale = (
            C.REWARD_SCORE_SCALE if reward_score_scale is None else reward_score_scale
        )
        self._reward_alive = C.REWARD_ALIVE if reward_alive is None else reward_alive
        self._reward_death = C.REWARD_DEATH if reward_death is None else reward_death
        self._reward_xpos = C.REWARD_XPOS if reward_xpos is None else reward_xpos
        self._x_front_frac = C.X_FRONT_FRAC if x_front_frac is None else x_front_frac
        self._reward_xmax = C.REWARD_XMAX if reward_xmax is None else reward_xmax
        self._reward_move_cost = (
            C.REWARD_MOVE_COST if reward_move_cost is None else reward_move_cost
        )
        self._reward_churn = C.REWARD_CHURN if reward_churn is None else reward_churn

    def _ram(self):
        return self.env.unwrapped.get_ram()

    def _read_powerups(self, ram):
        return {
            "powerbar": int(ram[C.ADDR_POWERBAR]),
            "missile": int(ram[C.ADDR_MISSILE]),
            "options": int(ram[C.ADDR_OPTIONS]),
            "shield": int(ram[C.ADDR_SHIELD]),
            "speed": int(ram[C.ADDR_SPEED]),
        }

    def _powerup_reward(self, ram):
        """Reward only INCREASES in power-up state, so upgrade caps self-enforce
        (a maxed value can't rise -> no reward -> no wasted-capsule incentive)."""
        cur, prev = self._read_powerups(ram), self._prev_pu
        r = 0.0
        if cur["powerbar"] > prev["powerbar"]:
            r += C.REWARD_CAPSULE * (cur["powerbar"] - prev["powerbar"])  # ate capsule(s)
        if cur["missile"] > prev["missile"]:
            r += C.REWARD_MISSILE
        if cur["options"] > prev["options"]:
            r += C.REWARD_OPTION * (cur["options"] - prev["options"])
        if cur["shield"] > prev["shield"]:
            r += C.REWARD_FORCEFIELD
        if cur["speed"] > prev["speed"]:
            # threshold: reward speed gained up to MAX_SPEED, heavily penalize beyond
            good = max(0, min(cur["speed"], C.MAX_SPEED) - min(prev["speed"], C.MAX_SPEED))
            over = max(0, cur["speed"] - max(prev["speed"], C.MAX_SPEED))
            r += C.REWARD_SPEED * good + C.REWARD_OVERSPEED * over
        self._prev_pu = cur
        return r

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram = self._ram()
        self._start_lives = int(ram[C.ADDR_LIVES])
        self._prev_lives = self._start_lives   # death = any DECREASE (a 1UP can raise
                                               # lives above start; `lives < start` then
                                               # misses the next death — measured: a free
                                               # respawn played 384 unarmed steps)
        self._start_stage = int(ram[C.ADDR_STAGE_NUM])
        self._start_vertical = int(ram[C.ADDR_STAGE_VERTICAL])
        self._cleared = False
        self._steps = 0
        self._prev_pu = self._read_powerups(ram)
        # running per-episode reward breakdown
        self._ep = {"score": 0.0, "alive": 0.0, "death": 0.0, "clear": 0.0,
                    "powerup": 0.0, "xpos": 0.0, "xmax": 0.0,
                    "move": 0.0, "churn": 0.0}
        self._prev_move = None    # movement index of the previous step (anti-jitter)
        # per-episode x diagnostics (true max, not terminal — see play.py mislabel).
        # x_pos is SCREEN x (15..232), a positioning proxy, not world progress.
        x0 = int(ram[C.ADDR_X_POS])
        self._x_max = x0
        self._x_sum = float(x0)
        self._x_count = 1
        # episode-local furthest-progress ratchet baseline (init to current x_frac)
        self._start_is_curr = bool(getattr(self.env.unwrapped, "_curriculum_start", False))
        self._start_label = getattr(self.env.unwrapped, "_curriculum_label", "level_start")
        self._best_x_frac = _x_frac(x0)
        return obs, self._augment(info, ram)

    def step(self, action):
        # `reward` from the inner env is the base score reward (scenario.json,
        # summed over the frame-skip). We split the total into named components.
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self._ram()
        lives = int(ram[C.ADDR_LIVES])
        self._steps += 1

        # x diagnostics: track true max / mean / terminal screen-x this episode
        x_now = int(ram[C.ADDR_X_POS])
        if x_now > self._x_max:
            self._x_max = x_now
        self._x_sum += x_now
        self._x_count += 1

        r_score = float(reward) * self._reward_score_scale
        r_alive = self._reward_alive
        r_death = 0.0
        r_clear = 0.0

        # 1) stay alive: per-step bonus, death penalty, end episode on death.
        # Death = lives DECREASED since last step (not `< start`: a score 1UP can
        # raise lives above start, and the next death would slip through — the
        # episode then continues on an unarmed respawn, inflating step counts).
        if lives < self._prev_lives:
            r_death = -self._reward_death
            info["life_lost"] = True
            if C.END_ON_LIFE_LOSS:
                terminated = True
        self._prev_lives = lives

        # 3) pass the level: detect Stage-1 -> Stage-2 transition.
        stage_changed = (
            int(ram[C.ADDR_STAGE_VERTICAL]) != self._start_vertical
            or int(ram[C.ADDR_STAGE_NUM]) != self._start_stage
        )
        if stage_changed and not self._cleared:
            self._cleared = True
            r_clear = C.REWARD_CLEAR
            info["stage_cleared"] = True
            self._capture_transition(ram)
            terminated = True  # Level 1 done; we start from Level 1 only for now

        # time limit (handled here so truncated episodes still report components)
        if self._steps >= C.MAX_EPISODE_STEPS:
            truncated = True

        # 2b) power-ups: eat capsules, accumulate, spend (Missile/Option/Force Field)
        r_powerup = self._powerup_reward(ram)

        # 2c) forward play (general; survival-arbitrated): a furthest-progress
        # ratchet to ESCAPE the camp-left basin + a mild front-quarter occupancy
        # bonus to HOLD position once there.
        x_frac = _x_frac(x_now)   # clamped to [0,1] (x_pos can momentarily exceed X_POS_MAX)

        # 2c-i) furthest-progress ratchet: pay only for NEW ground beyond this
        # episode's best (no retreat clawback, no front-camp bait). Implausibly
        # large positive jumps (screen wrap / death snap) advance best but pay 0.
        r_xmax = 0.0
        if self._reward_xmax:
            dx = x_frac - self._best_x_frac
            if dx > C.X_RATCHET_JUMP_CAP:
                self._best_x_frac = x_frac
            elif dx > 0.0:
                r_xmax = self._reward_xmax * dx
                self._best_x_frac = x_frac

        # 2c-ii) forward-position: mild per-step bonus for being in the FRONT quarter.
        r_xpos = 0.0
        if self._reward_xpos and x_frac >= self._x_front_frac:
            r_xpos = self._reward_xpos

        # 2d) anti-jitter: stillness is free, movement/vibration costs a little.
        # HOLD (move 0) fires while keeping position — in carve-a-path terrain
        # (e.g. the step-1736 web) staying on the cleared line is the skill.
        move = int(np.asarray(action).ravel()[0])
        r_move = -self._reward_move_cost if (self._reward_move_cost and move != 0) else 0.0
        r_churn = 0.0
        if self._reward_churn and self._prev_move is not None and move != self._prev_move:
            r_churn = -self._reward_churn
        self._prev_move = move

        total = (r_score + r_alive + r_death + r_clear + r_powerup + r_xpos
                 + r_xmax + r_move + r_churn)
        self._ep["score"] += r_score
        self._ep["alive"] += r_alive
        self._ep["death"] += r_death
        self._ep["clear"] += r_clear
        self._ep["powerup"] += r_powerup
        self._ep["xpos"] += r_xpos
        self._ep["xmax"] += r_xmax
        self._ep["move"] += r_move
        self._ep["churn"] += r_churn
        if terminated or truncated:
            info["reward_components"] = dict(self._ep)
            info["ep_steps"] = self._steps   # survival time = progress proxy (auto-scroller)
            info["max_x"] = self._x_max       # TRUE furthest screen-x reached
            info["terminal_x"] = x_now        # screen-x at death/timeout
            info["mean_x"] = self._x_sum / self._x_count
            info["curriculum_start"] = self._start_is_curr  # split metrics by start source
            info["start_state"] = self._start_label

        return obs, total, terminated, truncated, self._augment(info, ram)

    def _augment(self, info, ram):
        info = dict(info)
        info["x_pos"] = int(ram[C.ADDR_X_POS])
        info["y_pos"] = int(ram[C.ADDR_Y_POS])
        info["stage_num"] = int(ram[C.ADDR_STAGE_NUM])
        info["stage_vertical"] = int(ram[C.ADDR_STAGE_VERTICAL])
        return info

    def _capture_transition(self, ram):
        """Save the first observed stage-transition RAM — our elusive Stage-2
        reference. Diff against ram_dumps/stage1_baseline.npz to confirm which
        of ADDR_STAGE_NUM / ADDR_STAGE_VERTICAL is the true stage counter."""
        os.makedirs(C.RAM_DUMP_DIR, exist_ok=True)
        path = os.path.join(C.RAM_DUMP_DIR, f"stage_transition_pid{os.getpid()}.npz")
        if not os.path.exists(path):
            np.savez_compressed(path, ram=ram.copy(),
                                start_stage=self._start_stage,
                                start_vertical=self._start_vertical)
            print(f"[LifeForceWrapper] STAGE TRANSITION captured -> {path} "
                  f"(stage_num {self._start_stage}->{int(ram[C.ADDR_STAGE_NUM])}, "
                  f"vertical {self._start_vertical}->{int(ram[C.ADDR_STAGE_VERTICAL])})")


class FrameAudioRecorder(gym.Wrapper):
    """Capture every emulator frame's video + audio. Placed INSIDE the frame-skip
    so no frames/audio are dropped (the agent still decides once per skip; we just
    see all the in-between frames).

    Two uses:
      - store=True: buffer frames/audio in memory for a demo video with sound.
      - on_frame=fn: call fn(frame, audio) per frame for live playback (the
        callback can write audio to a sounddevice stream and draw the frame).
    """

    def __init__(self, env, on_frame=None, store=True):
        super().__init__(env)
        self.on_frame = on_frame
        self.store = store
        self.frames = []
        self.audio = []

    def step(self, action):
        out = self.env.step(action)
        frame = self.env.unwrapped.render()              # native RGB frame
        audio = self.env.unwrapped.em.get_audio().copy()  # (N, 2) int16
        if self.store:
            self.frames.append(frame)
            self.audio.append(audio)
        if self.on_frame is not None:
            self.on_frame(frame, audio)
        return out


def find_recorder(env):
    """Walk the wrapper chain to find the FrameAudioRecorder (if any)."""
    while env is not None:
        if isinstance(env, FrameAudioRecorder):
            return env
        env = getattr(env, "env", None)
    return None


def make_env(render_mode=None, preprocess=True, record_av=False, curriculum=False, seed=0,
             curriculum_glob=None, curriculum_mix=None, frame_skip=None,
             reward_score_scale=None, reward_alive=None, reward_death=None,
             reward_xpos=None, x_front_frac=None, reward_xmax=None,
             reward_move_cost=None, reward_churn=None):
    """Build one fully-wrapped Life Force env (a thunk-friendly constructor).

    record_av=True inserts a FrameAudioRecorder inside the frame-skip so play.py
    can write a video with sound. curriculum=True lets episodes start from saved
    states (for drilling hard sections); seed varies the per-env start-state
    sampling. curriculum_glob overrides which states are sampled (default: all of
    CURRICULUM_DIR/*.state) — e.g. "states/l3_wall.state" to drill ONE section.
    curriculum_mix overrides C.CURRICULUM_MIX (P(start from a curriculum state)).
    frame_skip overrides C.FRAME_SKIP for control-precision experiments while
    leaving the default training setup unchanged. reward_* overrides are optional
    experiment knobs; omitted values use config.py exactly.
    """
    env = retro.make(C.GAME, state=C.STATE, render_mode=render_mode)
    if curriculum:
        pattern = curriculum_glob or os.path.join(C.CURRICULUM_DIR, "*.state")
        mix = C.CURRICULUM_MIX if curriculum_mix is None else curriculum_mix
        env = CurriculumStart(env, pattern, mix, rng_seed=seed)
    env = Discretizer(env, C.MOVES, C.ACTIVATE_BUTTON)
    if record_av:
        env = FrameAudioRecorder(env)
    env = MaxAndSkipObservation(env, skip=C.FRAME_SKIP if frame_skip is None else frame_skip)
    env = LifeForceWrapper(
        env,
        reward_score_scale=reward_score_scale,
        reward_alive=reward_alive,
        reward_death=reward_death,
        reward_xpos=reward_xpos,
        x_front_frac=x_front_frac,
        reward_xmax=reward_xmax,
        reward_move_cost=reward_move_cost,
        reward_churn=reward_churn,
    )
    if preprocess:
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (C.FRAME_SIZE, C.FRAME_SIZE))
        env = FrameStackObservation(env, stack_size=C.FRAME_STACK)
    # NOTE: the episode time limit is enforced inside LifeForceWrapper (so it can
    # report reward components on truncation), not via a TimeLimit wrapper.
    return env


def make_thunk(seed=0, render_mode=None, curriculum=True,
               curriculum_glob=None, curriculum_mix=None, frame_skip=None,
               reward_score_scale=None, reward_alive=None, reward_death=None,
               reward_xpos=None, x_front_frac=None, reward_xmax=None,
               reward_move_cost=None, reward_churn=None):
    """Return a callable that builds a seeded env (for SB3 vec env constructors).
    Training envs use curriculum=True so they can start from saved hard-section
    states; seed varies both env seeding and curriculum sampling per env.
    curriculum_glob/curriculum_mix override which states are sampled and how often.
    frame_skip overrides C.FRAME_SKIP for that env only. reward_* overrides are
    passed into the worker env explicitly so SubprocVecEnv workers do not depend
    on parent-process config mutation."""
    def _init():
        env = make_env(render_mode=render_mode, curriculum=curriculum, seed=seed,
                       curriculum_glob=curriculum_glob, curriculum_mix=curriculum_mix,
                       frame_skip=frame_skip,
                       reward_score_scale=reward_score_scale,
                       reward_alive=reward_alive,
                       reward_death=reward_death,
                       reward_xpos=reward_xpos,
                       x_front_frac=x_front_frac,
                       reward_xmax=reward_xmax,
                       reward_move_cost=reward_move_cost,
                       reward_churn=reward_churn)
        env.reset(seed=seed)
        return env
    return _init
