"""Push-to-talk lip reading from the webcam.

SPACE  start / stop recording, then transcribe the clip
Q/ESC  quit

Usage:
    python push_to_talk.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth
"""

import argparse
import time

import cv2
import numpy as np

from vsr_pipeline import MODEL_FPS, VSRPipeline

MIN_SECONDS = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Push-to-talk visual speech recognition")
    parser.add_argument("--model", required=True, help="Path to a VSR .pth checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    return parser.parse_args()


def resample(frames, timestamps, fps=MODEL_FPS):
    """Pick the frames closest to a fixed fps grid, since webcams rarely run at exactly 25 fps."""
    timestamps = np.asarray(timestamps) - timestamps[0]
    targets = np.arange(0, timestamps[-1], 1.0 / fps)
    idx = np.clip(np.searchsorted(timestamps, targets), 0, len(frames) - 1)
    return np.stack([frames[i] for i in idx])


def draw(frame, recording, message):
    view = cv2.flip(frame, 1)  # mirror for display only; the model sees the unflipped frames
    if recording:
        cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(view, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(view, "SPACE: record   Q: quit", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if message:
        h = view.shape[0]
        cv2.rectangle(view, (0, h - 50), (view.shape[1], h), (0, 0, 0), -1)
        cv2.putText(view, message, (15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return view


def main():
    args = parse_args()

    print("Loading model...")
    pipeline = VSRPipeline(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS, allow camera access for your terminal app.")

    window = "auto_avsr push-to-talk"
    recording, frames, timestamps, message = False, [], [], ""
    print("Ready. Press SPACE to start/stop recording, Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if recording:
            frames.append(frame)
            timestamps.append(time.monotonic())

        cv2.imshow(window, draw(frame, recording, message))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key != ord(" "):
            continue

        if not recording:
            recording, frames, timestamps, message = True, [], [], ""
            continue

        recording = False
        if len(frames) < 2 or timestamps[-1] - timestamps[0] < MIN_SECONDS:
            message = "Clip too short, try again."
            continue

        cv2.imshow(window, draw(frame, False, "Transcribing..."))
        cv2.waitKey(1)

        clip = resample(frames, timestamps)
        clip = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip])
        start = time.monotonic()
        try:
            transcript = pipeline.transcribe(clip)
            message = transcript or "(empty transcript)"
            print(f"[{time.monotonic() - start:.1f}s] {message}")
        except Exception as e:  # e.g. no face found, mouth too close to the frame edge
            message = f"Failed: {e}"
            print(message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
