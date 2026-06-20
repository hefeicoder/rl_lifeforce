"""Watch / evaluate a trained Life Force agent.

Default is a live 3x window with game sound. Modes:
  (default)        : live OpenCV window + sound (== --render human --audio --scale 3)
  --render video   : record an MP4 instead (keeps sound; --no-audio for silent)
  --no-audio       : disable sound

Live sound uses sounddevice in blocking-write mode: the audio buffer paces the
loop to real time, so video stays synced. It needs the machine mostly to itself
(a busy CPU causes audio-underrun clicks) — stop training first.

Usage:
  python -m src.play --model checkpoints/lifeforce_ppo_final.zip             # live + sound
  python -m src.play --model ... --render human --no-audio                   # live, silent
  python -m src.play --model ... --render video                             # mp4 with sound
  python -m src.play --model ... --render video --no-audio                  # silent mp4
"""
import argparse
import os
import subprocess
import tempfile
import wave

import cv2
import imageio
import imageio_ffmpeg
import numpy as np
from stable_baselines3 import PPO

from . import config as C
from .env import make_env, find_recorder

WINDOW = "Life Force - agent"


def _display_size(frame, scale, aspect):
    """Target (w, h): scale the native frame, then stretch width to `aspect`
    (NES looks right at 4:3, not at its near-square framebuffer ratio)."""
    h, w = frame.shape[:2]
    out_h = h * scale
    out_w = int(round(out_h * aspect)) if aspect else w * scale
    return out_w, out_h


def _write_video_with_audio(frames, audio_chunks, rate, out, fps=60):
    """Mux full-rate frames + captured audio into an MP4 via ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.mkdtemp()
    vpath, apath = os.path.join(tmp, "v.mp4"), os.path.join(tmp, "a.wav")
    imageio.mimsave(vpath, [np.asarray(f) for f in frames], fps=fps)
    samples = np.concatenate(audio_chunks, axis=0).astype("<i2")
    with wave.open(apath, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(round(rate)))
        w.writeframes(samples.tobytes())
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", vpath, "-i", apath,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to a trained .zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--render", choices=["video", "human"], default="human",
                   help="'human' = live window (default); 'video' = record an mp4")
    p.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True,
                   help="game sound (default on; --no-audio to disable)")
    p.add_argument("--fps", type=int, default=30, help="pacing for live --render human")
    p.add_argument("--scale", type=int, default=3, help="live window size = native x scale")
    p.add_argument("--aspect", type=float, default=4 / 3,
                   help="live window width:height ratio (0 = keep native, ~square)")
    p.add_argument("--out", default=os.path.join(C.VIDEO_DIR, "play.mp4"))
    args = p.parse_args()

    live = args.render == "human"
    record_av = args.audio          # capture every frame's video+audio when sound is wanted
    env = make_env(render_mode="rgb_array", record_av=record_av)
    model = PPO.load(args.model)
    recorder = find_recorder(env)

    # Live + sound: stream audio (blocking write paces to real-time) and draw each
    # frame from the recorder's per-frame hook.
    audio_stream = None
    quit_flag = {"q": False}
    if live and args.audio:
        import sounddevice as sd
        rate = int(round(env.unwrapped.em.get_audio_rate()))
        audio_stream = sd.OutputStream(samplerate=rate, channels=2, dtype="int16")
        audio_stream.start()

        def _on_frame(frame, audio):
            audio_stream.write(audio)          # blocks ~1 frame -> real-time clock + sound
            w, h = _display_size(frame, args.scale, args.aspect)
            disp = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(WINDOW, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_flag["q"] = True

        recorder.on_frame = _on_frame
        recorder.store = False                 # stream live; don't buffer in memory

    # which path renders inside the loop (the recorder hook handles live+audio)
    loop_renders = (live and not args.audio) or (not live and not record_av)

    frames, cleared_count = [], 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward, steps = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, info = env.step(action)   # MultiDiscrete: [move, activate]
            if loop_renders:
                frame = env.unwrapped.render()
                if live:                       # live, no sound: draw every 4th frame
                    w, h = _display_size(frame, args.scale, args.aspect)
                    disp = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(WINDOW, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF == ord("q"):
                        done = True
                else:                          # silent video: buffer agent-step frames
                    frames.append(frame)
            ep_reward += reward
            steps += 1
            done = done or term or trunc or quit_flag["q"]
        cleared = info.get("stage_cleared", False)
        cleared_count += int(cleared)
        print(f"ep {ep}: score={info.get('score')} steps={steps} "
              f"reward={ep_reward:.1f} max_x={info.get('x_pos')} "
              f"{'CLEARED LEVEL 1' if cleared else 'did not clear'}")
        if quit_flag["q"]:
            break

    print(f"\ncleared {cleared_count}/{args.episodes} episodes")
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
    if live:
        cv2.destroyAllWindows()
    else:
        os.makedirs(C.VIDEO_DIR, exist_ok=True)
        if record_av:
            rate = env.unwrapped.em.get_audio_rate()
            _write_video_with_audio(recorder.frames, recorder.audio, rate, args.out)
            print(f"video (with sound) -> {args.out}")
        else:
            imageio.mimsave(args.out, [np.asarray(f) for f in frames], fps=30)
            print(f"video -> {args.out}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
