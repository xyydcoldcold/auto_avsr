"""Run the VSR model over every clip in the manifest and save the n-best hypotheses.

Usage (from the repo root):
    python silent_speech/infer.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth
    python silent_speech/evaluate.py --pred silent_speech/outputs/predictions.jsonl \
        --manifest silent_speech/data/manifest.jsonl

Writes one line per clip in the format evaluate.py expects:
    {"clip_id": "...", "nbest": [{"text": "...", "score": -3.2}, ...],
     "latency_ms": 840, "crop_ms": 300, "decode_ms": 540, "beam_size": 10,
     "model": "vsr_trlrs2lrs3vox2avsp_base", "adapter": null}

latency_ms covers mouth cropping + decoding, not reading the file from disk.
A clip that fails (e.g. no face found) gets a single empty hypothesis and an "error"
field, so it counts as all-deletions in the WER instead of silently dropping out.
Re-running appends only the clips missing from the output file.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torchvision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for vsr_pipeline and the upstream modules

from vsr_pipeline import MODEL_FPS, VSRPipeline  # noqa: E402


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_frames(path):
    frames, _, info = torchvision.io.read_video(str(path), pts_unit="sec")
    frames = frames.numpy()
    fps = info.get("video_fps", MODEL_FPS)
    if abs(fps - MODEL_FPS) > 0.5:  # record.py already saves 25 fps; this covers other clips
        idx = np.round(np.arange(0, len(frames) * MODEL_FPS / fps) * fps / MODEL_FPS).astype(int)
        frames = frames[np.clip(idx, 0, len(frames) - 1)]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a VSR .pth checkpoint")
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.jsonl"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "predictions.jsonl"))
    ap.add_argument("--beam-size", type=int, default=10)
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N pending clips")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    clips = load_jsonl(manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {p["clip_id"] for p in load_jsonl(out)}
    pending = [c for c in clips if c["clip_id"] not in done][: args.limit]
    print(f"{len(clips)} clips in manifest, {len(done)} already in {out}, running {len(pending)}")
    if not pending:
        return

    model_name = Path(args.model).stem
    print("Loading model...")
    pipeline = VSRPipeline(args.model)

    failed = 0
    with open(out, "a", encoding="utf-8") as f:
        for n, clip in enumerate(pending, 1):
            pred = {"clip_id": clip["clip_id"]}
            try:
                frames = read_frames(manifest.parent / clip["path"])
                start = time.perf_counter()
                hyps, timings = pipeline.transcribe_nbest(frames, beam_size=args.beam_size, nbest=args.nbest)
                pred.update(nbest=hyps or [{"text": "", "score": None}],
                            latency_ms=round((time.perf_counter() - start) * 1000), **timings)
            except Exception as e:  # e.g. no face found, mouth too close to the frame edge
                failed += 1
                pred.update(nbest=[{"text": "", "score": None}], latency_ms=0, error=str(e))
            pred.update(beam_size=args.beam_size, model=model_name, adapter=None)
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()

            top = pred["nbest"][0]["text"]
            status = f"ERROR {pred['error']}" if "error" in pred else f"{pred['latency_ms']} ms  {top!r}"
            print(f"[{n}/{len(pending)}] {clip['clip_id']}  ref={clip['text']!r}  {status}")

    if failed:
        print(f"{failed} clip(s) failed and were scored as empty transcripts.")


if __name__ == "__main__":
    main()
