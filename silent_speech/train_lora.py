"""Train a personal LoRA adapter on silent training clips (the backbone stays frozen).

Usage (from the repo root):
    python silent_speech/train_lora.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth --device mps
    python silent_speech/train_lora.py --model ... --clips 100     # only the first 100 clips
    python silent_speech/train_lora.py --model ... --minutes 5     # or the first 5 minutes of clips

Data: silent_speech/data_train/manifest.jsonl (recorded with record.py --data-dir .../data_train).
Clips are taken in prompt order (t000, t001, ...), so --minutes N is always the same subset; that
gives the "minutes of personal data -> WER" curve. Mouth crops are cached in data_train/crops/.

Training: LoRA (rank 8) on the encoder's attention projections, loss = the model's own
0.1 * CTC + 0.9 * attention loss, training-time augmentation from upstream (random crop,
time masking). A held-out slice of the training clips (never the test set) picks the best epoch.

Output: silent_speech/adapters/lora_<100clips|5min|all>.pt with the LoRA weights and the config; pass it to
infer.py / closed_set.py with --adapter.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# The visual frontend's 3D max-pool has no MPS kernel; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from datamodule.transforms import VideoTransform  # noqa: E402
from infer import load_jsonl, read_frames  # noqa: E402
from vsr_pipeline import MODEL_FPS, VSRPipeline, add_lora, lora_state_dict  # noqa: E402


def select_clips(manifest, minutes=None, n_clips=None):
    clips = sorted(manifest, key=lambda c: c["prompt_id"])
    if n_clips is not None:
        return clips[:n_clips]
    if minutes is None:
        return clips
    picked, total = [], 0.0
    for c in clips:
        if total >= minutes * 60:
            break
        picked.append(c)
        total += c["n_frames"] / MODEL_FPS
    return picked


def load_crops(pipeline, clips, data_dir):
    """Mouth crops (uint8 T x 3 x 96 x 96), cached per clip since face tracking is the slow part."""
    cache = data_dir / "crops"
    cache.mkdir(exist_ok=True)
    crops = {}
    for n, c in enumerate(clips, 1):
        path = cache / f"{c['clip_id']}.pt"
        if path.exists():
            crops[c["clip_id"]] = torch.load(path, weights_only=True)
            continue
        try:
            crops[c["clip_id"]] = pipeline.crop(read_frames(data_dir / c["path"]))
            torch.save(crops[c["clip_id"]], path)
        except Exception as e:  # e.g. no face found
            print(f"skip {c['clip_id']}: {e}")
        if n % 50 == 0:
            print(f"cropped {n}/{len(clips)}")
    return crops


def make_batch(items, transform, tokenize, device):
    videos = [transform(crop) for crop, _ in items]
    labels = [tokenize(text) for _, text in items]
    lengths = torch.tensor([len(v) for v in videos])
    x = torch.zeros(len(videos), int(lengths.max()), *videos[0].shape[1:])
    for i, v in enumerate(videos):
        x[i, : len(v)] = v
    y = torch.full((len(labels), max(len(l) for l in labels)), -1, dtype=torch.long)
    for i, l in enumerate(labels):
        y[i, : len(l)] = l
    return x.to(device), lengths.to(device), y.to(device)


def run_epoch(model, data, batch_size, transform, tokenize, device, optimizer=None):
    total, count = 0.0, 0
    for i in range(0, len(data), batch_size):
        x, lengths, y = make_batch(data[i : i + batch_size], transform, tokenize, device)
        with torch.set_grad_enabled(optimizer is not None):
            loss, _, _, _ = model(x, lengths, y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss) * len(x)
        count += len(x)
        del x, lengths, y, loss
        if device == "mps":
            torch.mps.empty_cache()  # every batch has a new shape; cached buffers otherwise pile up and swap
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the base VSR .pth checkpoint")
    ap.add_argument("--data-dir", default=str(HERE / "data_train"))
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--clips", type=int, default=None, help="Use only the first N clips (in prompt order)")
    group.add_argument("--minutes", type=float, default=None, help="Use only the first N minutes of clips")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Training clips held out to pick the best epoch")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Default: silent_speech/adapters/lora_<minutes>min.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    clips = select_clips(load_jsonl(data_dir / "manifest.jsonl"), args.minutes, args.clips)
    if args.clips and len(clips) < args.clips:
        raise SystemExit(f"Only {len(clips)} training clips recorded, asked for {args.clips}")
    minutes = sum(c["n_frames"] for c in clips) / MODEL_FPS / 60
    print(f"{len(clips)} training clips, {minutes:.1f} min")

    pipeline = VSRPipeline(args.model)  # CPU while cropping
    crops = load_crops(pipeline, clips, data_dir)
    data = [(crops[c["clip_id"]], c["text"]) for c in clips if c["clip_id"] in crops]
    random.shuffle(data)
    n_val = int(len(data) * args.val_frac)
    val, train = data[:n_val], data[n_val:]
    print(f"train {len(train)} clips, val {len(val)} clips")

    model = pipeline.modelmodule.model
    params = add_lora(model, args.rank, args.alpha)
    model.to(args.device)
    print(f"LoRA params: {sum(p.numel() for p in params):,} "
          f"({100 * sum(p.numel() for p in params) / sum(p.numel() for p in model.parameters()):.2f}% of the model)")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    tokenize = lambda text: pipeline.tokenize(text)  # noqa: E731
    train_tf, val_tf = VideoTransform("train"), VideoTransform("test")

    size = f"{args.clips}clips" if args.clips else ("all" if args.minutes is None else f"{args.minutes:g}min")
    out = Path(args.out) if args.out else HERE / "adapters" / f"lora_{size}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"rank": args.rank, "alpha": args.alpha, "targets": "encoder attention q/k/v/out",
              "base_model": Path(args.model).stem, "clips": len(clips), "minutes": round(minutes, 2), "train_clips": len(train),
              "val_clips": len(val), "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "device": args.device}

    best, log = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        model.frontend.eval()  # frozen BatchNorm statistics stay as pretrained
        random.shuffle(train)
        train_loss = run_epoch(model, train, args.batch_size, train_tf, tokenize, args.device, optimizer)
        model.eval()
        val_loss = run_epoch(model, val, args.batch_size, val_tf, tokenize, args.device) if val else train_loss
        log.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})
        mark = ""
        if val_loss < best:
            best = val_loss
            torch.save({"config": {**config, "best_epoch": epoch}, "lora": lora_state_dict(model)}, out)
            mark = "  saved"
        mem = f"  gpu {torch.mps.driver_allocated_memory() / 2**30:.1f} GB" if args.device == "mps" else ""
        print(f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f}  {time.time() - start:.0f}s{mem}{mark}")

    (out.with_suffix(".json")).write_text(json.dumps({"config": config, "log": log}, indent=2))
    print(f"best val loss {best:.3f} -> {out}")


if __name__ == "__main__":
    main()
