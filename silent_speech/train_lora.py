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
time masking).

Validation split: a fixed ~10% of the training clips (never the test set), chosen by a hash of the
prompt id, so it does NOT change with --seed and is nested across sizes (val of 300 clips is inside
val of 400 clips). --seed now only changes LoRA init, batch order and augmentation.

Weights: the saved adapter is the average of the last --avg-last epochs (default 5), not the single
epoch with the lowest val loss; picking one epoch on a ~30-clip val set mostly picks noise. The average
is exact: mean_i(B_i A_i) is stored as one LoRA of rank K*r, so infer.py / closed_set.py load it unchanged.
The lowest-val-loss epoch is still saved next to it as <name>_best.pt, for comparison.

Output: silent_speech/adapters/lora_<300clips|5min|all>_avg_s<seed>.pt (+ _best.pt, + .json log);
pass it to infer.py / closed_set.py with --adapter. To judge a change, train >= 2 seeds per setting and
compare the predictions with compare.py (see run_seeds.sh).
"""
import argparse
import hashlib
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


def is_val(prompt_id, val_frac):
    """Fixed held-out split: a clip is in val iff a hash of its prompt id falls below val_frac.
    Independent of --seed, and nested across --clips sizes."""
    return int(hashlib.md5(prompt_id.encode()).hexdigest(), 16) / 16**32 < val_frac


def average_lora(snapshots):
    """Exact average of K LoRA updates: mean_i(B_i A_i) = [B_1 .. B_K] / K @ [A_1; ..; A_K],
    i.e. one LoRA of rank K*r with the same scale. (Averaging A and B separately would not be exact.)"""
    k, out = len(snapshots), {}
    for name in snapshots[0]:
        if name.endswith("lora_A"):
            out[name] = torch.cat([s[name] for s in snapshots], dim=0)
        elif name.endswith("lora_B"):
            out[name] = torch.cat([s[name] for s in snapshots], dim=1) / k
    return out


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
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fixed share of training clips held out (hash of prompt id)")
    ap.add_argument("--avg-last", type=int, default=5, help="Average the LoRA weights of the last K epochs (1 = last epoch only)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0, help="LoRA init, batch order, augmentation (not the val split)")
    ap.add_argument("--out", default=None, help="Default: silent_speech/adapters/lora_<size>_avg_s<seed>.pt")
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
    kept = [c for c in clips if c["clip_id"] in crops]
    val = [(crops[c["clip_id"]], c["text"]) for c in kept if is_val(c["prompt_id"], args.val_frac)]
    train = [(crops[c["clip_id"]], c["text"]) for c in kept if not is_val(c["prompt_id"], args.val_frac)]
    print(f"train {len(train)} clips, val {len(val)} clips (fixed split)")
    avg_last = max(1, min(args.avg_last, args.epochs))

    model = pipeline.modelmodule.model
    params = add_lora(model, args.rank, args.alpha)
    model.to(args.device)
    print(f"LoRA params: {sum(p.numel() for p in params):,} "
          f"({100 * sum(p.numel() for p in params) / sum(p.numel() for p in model.parameters()):.2f}% of the model)")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    tokenize = lambda text: pipeline.tokenize(text)  # noqa: E731
    train_tf, val_tf = VideoTransform("train"), VideoTransform("test")

    size = f"{args.clips}clips" if args.clips else ("all" if args.minutes is None else f"{args.minutes:g}min")
    out = Path(args.out) if args.out else HERE / "adapters" / f"lora_{size}_avg_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_best = out.with_name(f"{out.stem}_best.pt")
    config = {"rank": args.rank, "alpha": args.alpha, "targets": "encoder attention q/k/v/out",
              "base_model": Path(args.model).stem, "clips": len(clips), "minutes": round(minutes, 2), "train_clips": len(train),
              "val_clips": len(val), "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "device": args.device,
              "seed": args.seed, "val_split": f"md5(prompt_id) < {args.val_frac}"}

    best, best_epoch, log, snapshots = float("inf"), None, [], []
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
            best, best_epoch = val_loss, epoch
            torch.save({"config": {**config, "best_epoch": epoch}, "lora": lora_state_dict(model)}, out_best)
            mark = "  best"
        if epoch > args.epochs - avg_last:
            snapshots.append(lora_state_dict(model))
            mark += "  (averaged)"
        mem = f"  gpu {torch.mps.driver_allocated_memory() / 2**30:.1f} GB" if args.device == "mps" else ""
        print(f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f}  {time.time() - start:.0f}s{mem}{mark}")

    # One LoRA of rank K*r; the pipeline reads rank/alpha from the config, so keep alpha/rank = the trained scale.
    k = len(snapshots)
    avg_epochs = list(range(args.epochs - k + 1, args.epochs + 1))
    avg_config = {**config, "rank": args.rank * k, "alpha": args.alpha * k, "base_rank": args.rank, "avg_epochs": avg_epochs}
    torch.save({"config": avg_config, "lora": average_lora(snapshots)}, out)

    summary = {"config": avg_config, "best_epoch": best_epoch, "best_val_loss": round(best, 4), "log": log}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"averaged epochs {avg_epochs[0]}-{avg_epochs[-1]} -> {out}")
    print(f"best single epoch {best_epoch} (val {best:.3f}) -> {out_best}")


if __name__ == "__main__":
    main()
