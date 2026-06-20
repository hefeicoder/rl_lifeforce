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
        self.action_space = gym.spaces.MultiDiscrete([len(self._moves), 2])

    def action(self, act):
        move_idx, activate = int(act[0]), int(act[1])
        arr = self._moves[move_idx].copy()
        if activate:
            arr[self._activate_idx] = 1
        return arr


class LifeForceWrapper(gym.Wrapper):
    """RAM-based reward shaping + episode logic for Life Force.

    Reads the addresses found during the RAM hunt (see docs/ram_map.md) and
    encodes the project's objective: stay alive, score, pass the level. Also
    auto-captures the Stage-1 -> Stage-2 transition RAM the first time it is
    seen, which is how we finish confirming the stage-clear detector.
    """

    def __init__(self, env):
        super().__init__(env)
        self._captured = False

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
            r += C.REWARD_SPEED * (cur["speed"] - prev["speed"])
        self._prev_pu = cur
        return r

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram = self._ram()
        self._start_lives = int(ram[C.ADDR_LIVES])
        self._start_stage = int(ram[C.ADDR_STAGE_NUM])
        self._start_vertical = int(ram[C.ADDR_STAGE_VERTICAL])
        self._cleared = False
        self._steps = 0
        self._prev_pu = self._read_powerups(ram)
        # running per-episode reward breakdown
        self._ep = {"score": 0.0, "alive": 0.0, "death": 0.0, "clear": 0.0, "powerup": 0.0}
        return obs, self._augment(info, ram)

    def step(self, action):
        # `reward` from the inner env is the base score reward (scenario.json,
        # summed over the frame-skip). We split the total into named components.
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self._ram()
        lives = int(ram[C.ADDR_LIVES])
        self._steps += 1

        r_score = float(reward) * C.REWARD_SCORE_SCALE
        r_alive = C.REWARD_ALIVE
        r_death = 0.0
        r_clear = 0.0

        # 1) stay alive: per-step bonus, death penalty, end episode on death.
        if lives < self._start_lives:
            r_death = -C.REWARD_DEATH
            info["life_lost"] = True
            if C.END_ON_LIFE_LOSS:
                terminated = True

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

        total = r_score + r_alive + r_death + r_clear + r_powerup
        self._ep["score"] += r_score
        self._ep["alive"] += r_alive
        self._ep["death"] += r_death
        self._ep["clear"] += r_clear
        self._ep["powerup"] += r_powerup
        if terminated or truncated:
            info["reward_components"] = dict(self._ep)

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


def make_env(render_mode=None, preprocess=True, record_av=False):
    """Build one fully-wrapped Life Force env (a thunk-friendly constructor).

    record_av=True inserts a FrameAudioRecorder inside the frame-skip so play.py
    can write a video with sound.
    """
    env = retro.make(C.GAME, state=C.STATE, render_mode=render_mode)
    env = Discretizer(env, C.MOVES, C.ACTIVATE_BUTTON)
    if record_av:
        env = FrameAudioRecorder(env)
    env = MaxAndSkipObservation(env, skip=C.FRAME_SKIP)
    env = LifeForceWrapper(env)
    if preprocess:
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (C.FRAME_SIZE, C.FRAME_SIZE))
        env = FrameStackObservation(env, stack_size=C.FRAME_STACK)
    # NOTE: the episode time limit is enforced inside LifeForceWrapper (so it can
    # report reward components on truncation), not via a TimeLimit wrapper.
    return env


def make_thunk(seed=0, render_mode=None):
    """Return a callable that builds a seeded env (for SB3 vec env constructors)."""
    def _init():
        env = make_env(render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init
