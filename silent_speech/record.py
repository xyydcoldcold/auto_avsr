"""Record the fixed test set from the webcam, one prompt at a time.

Usage (from the repo root):
    python silent_speech/record.py --session day1 --condition silent --camera 1
    python silent_speech/record.py --session day1 --condition voiced --camera 1

Keys (in the preview window):
    SPACE  start / stop recording the current prompt
    R      redo the last saved clip
    S      skip this prompt
    Q      quit (progress is kept; re-running skips prompts already recorded)

Output (clips are resampled to the 25 fps the model was trained on):
    silent_speech/data/raw/<session>/<clip_id>.mp4
    silent_speech/data/manifest.jsonl   (one JSON line per clip, "path" relative to data/)
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST = DATA / "manifest.jsonl"
MODEL_FPS = 25
MIN_SECONDS = 0.5


def load_prompts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["text"].strip()]


def recorded_ids():
    if not MANIFEST.exists():
        return set()
    with open(MANIFEST, encoding="utf-8") as f:
        return {json.loads(line)["clip_id"] for line in f if line.strip()}


def resample(frames, timestamps, fps=MODEL_FPS):
    """Pick the frames closest to a fixed fps grid, since webcams rarely run at exactly 25 fps."""
    timestamps = np.asarray(timestamps) - timestamps[0]
    targets = np.arange(0, timestamps[-1], 1.0 / fps)
    idx = np.clip(np.searchsorted(timestamps, targets), 0, len(frames) - 1)
    return [frames[i] for i in idx]


def draw(frame, lines, recording):
    out = cv2.flip(frame, 1)  # mirror for display only; the saved clip is unflipped
    h, w = out.shape[:2]
    # Guide box: keep your mouth roughly inside it.
    cv2.rectangle(out, (w // 2 - 120, h // 2), (w // 2 + 120, h // 2 + 160), (0, 255, 0), 1)
    # Plain white text on a dark banner (an outlined text reads as doubled when the window is scaled).
    banner_h = 20 + sum(int(40 * scale + 10) for _, scale in lines)
    out[:banner_h] = (out[:banner_h] * 0.3).astype(out.dtype)
    y = 40
    for text, scale in lines:
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += int(40 * scale + 10)
    if recording:
        cv2.circle(out, (w - 40, 40), 15, (0, 0, 255), -1)
    return out


def drop_last_manifest_line():
    lines = MANIFEST.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines:
        last = json.loads(lines[-1])
        (DATA / last["path"]).unlink(missing_ok=True)
        MANIFEST.write_text("".join(lines[:-1]), encoding="utf-8")
        print(f"removed {last['clip_id']}, re-record it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(HERE / "prompts.csv"))
    ap.add_argument("--session", required=True, help="e.g. day1, day2 (used for cross-day tests)")
    ap.add_argument("--condition", default="silent", choices=["silent", "voiced"])
    ap.add_argument("--camera", type=int, default=0, help="Camera index (1 on Ethan's Mac; 0 is Continuity Camera)")
    args = ap.parse_args()

    out_dir = DATA / "raw" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try another --camera index, or check macOS camera permission.")

    done = recorded_ids()
    prompts = load_prompts(args.prompts)
    queue = [p for p in prompts if f"{args.session}_{args.condition}_{p['id']}" not in done]
    print(f"{len(queue)} prompts left for session={args.session} condition={args.condition}")

    i, last_saved, note = 0, None, ""
    while i < len(queue):
        p = queue[i]
        clip_id = f"{args.session}_{args.condition}_{p['id']}"
        mode = "SILENT: move lips, no sound" if args.condition == "silent" else "VOICED: speak normally"
        header = [(f"[{i + 1}/{len(queue)}] {p['split']}  |  {mode}", 0.6),
                  (p["text"], 1.0),
                  (note or "SPACE start/stop   R redo last   S skip   Q quit", 0.5)]

        # Wait for SPACE to start.
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow("record", draw(frame, header, False))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord(" "), ord("s"), ord("q"), ord("r")):
                break
        note = ""
        if k == ord("q"):
            break
        if k == ord("s"):
            i += 1
            continue
        if k == ord("r"):
            if last_saved is not None:
                drop_last_manifest_line()
                i = last_saved
                last_saved = None
            continue

        # Record until SPACE again.
        frames, timestamps = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(frame)
            timestamps.append(time.monotonic())
            cv2.imshow("record", draw(frame, header, True))
            if (cv2.waitKey(1) & 0xFF) == ord(" "):
                break
        duration = timestamps[-1] - timestamps[0]
        if len(frames) < 2 or duration < MIN_SECONDS:
            note = "Clip too short, press SPACE to try again."
            continue
        camera_fps = (len(frames) - 1) / duration
        clip = resample(frames, timestamps)

        path = out_dir / f"{clip_id}.mp4"
        h, w = clip[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), MODEL_FPS, (w, h))
        for fr in clip:
            writer.write(fr)
        writer.release()

        rec = {
            "clip_id": clip_id,
            "prompt_id": p["id"],
            "text": p["text"],
            "split": p["split"],              # in_scene / open / out_scene
            "scene": p.get("scene", ""),
            "condition": args.condition,      # silent / voiced
            "session": args.session,
            "path": str(path.relative_to(DATA)),
            "fps": MODEL_FPS,
            "camera_fps": round(camera_fps, 2),
            "n_frames": len(clip),
            "duration_s": round(duration, 2),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(MANIFEST, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"saved {clip_id}  {len(clip)} frames ({camera_fps:.1f} fps from camera)")
        last_saved = i
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
