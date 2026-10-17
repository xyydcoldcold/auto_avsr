"""Closed phrase set check: pick the answer from a fixed phrase list instead of decoding freely.

Usage (from the repo root):
    python silent_speech/closed_set.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth

For every clip, the model scores every phrase in phrases.txt (teacher forcing: decoder + CTC,
weighted 0.9 / 0.1 like the beam search). Scores are cached in outputs/phrase_scores.jsonl,
so re-runs only score new clips. Then, for each list size N, it reports:

    top1 / top3   in-scene clips: the right phrase is ranked 1st / in the top 3
    FA@FR20       open + out-of-scene clips (not in the list) accepted as some phrase, with the
                  reject threshold set so 20% of in-scene clips are rejected
    ms            time to score N phrases for one clip (encoding not included), scaled from the
                  timed full list since scoring time grows linearly with the number of phrases

Lists of size N always contain the 50 in-scene test prompts plus N-50 random other phrases,
averaged over several random draws. The threshold is set on the same clips it is scored on,
so FA is optimistic until it is checked on another session.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from infer import load_jsonl, read_frames  # noqa: E402

CTC_WEIGHT = 0.1
SIZES = [50, 100, 250, 500]
SEEDS = 5
FALSE_REJECT = 0.2


def norm(s):
    return " ".join(s.lower().split())


def score_clips(args, phrases, clips, cache_path):
    from vsr_pipeline import VSRPipeline

    done = {r["clip_id"] for r in load_jsonl(cache_path) if r["phrases"] == args.list_id}
    pending = [c for c in clips if c["clip_id"] not in done]
    print(f"{len(done)} clips already scored, scoring {len(pending)}")
    if not pending:
        return

    pipeline = VSRPipeline(args.model)
    tokens = [pipeline.tokenize(p) for p in phrases]
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, clip in enumerate(pending, 1):
            enc = pipeline.encode(read_frames(Path(args.manifest).parent / clip["path"]))
            start = time.perf_counter()
            att, ctc = pipeline.score_phrases(enc, tokens)
            ms = round((time.perf_counter() - start) * 1000)
            f.write(json.dumps({"clip_id": clip["clip_id"], "phrases": args.list_id, "att": att,
                                "ctc": [max(c, -1e9) for c in ctc], "ms": ms}) + "\n")
            f.flush()
            print(f"[{n}/{len(pending)}] {clip['clip_id']}  {ms} ms")


def scores(row, n_tokens, length_norm):
    s = (1 - CTC_WEIGHT) * np.array(row["att"]) + CTC_WEIGHT * np.array(row["ctc"])
    return s / (n_tokens + 1) if length_norm else s  # +1 for <eos>


def report(phrases, clips, cached, n_tokens):
    index = {p: i for i, p in enumerate(phrases)}
    test_ids = [index[norm(c["text"])] for c in clips if c["split"] == "in_scene"]
    test_ids = sorted(set(test_ids))
    others = [i for i in range(len(phrases)) if i not in set(test_ids)]

    print(f"\n{'condition':10} {'score':10} {'N':>4} {'top1':>6} {'top3':>6} {'FA@FR20':>8} {'ms':>6}")
    for condition in ("silent", "voiced"):
        group = [c for c in clips if c["condition"] == condition and c["clip_id"] in cached]
        pos = [c for c in group if c["split"] == "in_scene"]
        neg = [c for c in group if c["split"] != "in_scene"]
        if not pos:
            continue
        ms = np.mean([cached[c["clip_id"]]["ms"] for c in group])
        for length_norm in (False, True):
            for size in SIZES:
                if size > len(phrases):
                    continue
                seeds = 1 if size == len(test_ids) else SEEDS
                top1, top3, fa = [], [], []
                for seed in range(seeds):
                    subset = np.array(test_ids + random.Random(seed).sample(others, size - len(test_ids)))

                    def ranked(clip):
                        s = scores(cached[clip["clip_id"]], n_tokens, length_norm)[subset]
                        order = np.argsort(-s)
                        return subset[order], s[order[0]] - s[order[1]]  # margin = confidence

                    pos_rank = [ranked(c) for c in pos]
                    gold = [index[norm(c["text"])] for c in pos]
                    top1.append(np.mean([r[0] == g for (r, _), g in zip(pos_rank, gold)]))
                    top3.append(np.mean([g in r[:3] for (r, _), g in zip(pos_rank, gold)]))
                    threshold = np.quantile([m for _, m in pos_rank], FALSE_REJECT)
                    fa.append(np.mean([ranked(c)[1] >= threshold for c in neg]) if neg else float("nan"))
                print(f"{condition:10} {'per-token' if length_norm else 'sum':10} {size:>4} "
                      f"{np.mean(top1):>6.1%} {np.mean(top3):>6.1%} {np.mean(fa):>8.1%} {ms * size / len(phrases):>6.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a VSR .pth checkpoint")
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.jsonl"))
    ap.add_argument("--phrases", default=str(HERE / "phrases.txt"))
    ap.add_argument("--cache", default=str(HERE / "outputs" / "phrase_scores.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N clips (for testing)")
    args = ap.parse_args()

    phrases = [norm(p) for p in Path(args.phrases).read_text(encoding="utf-8").splitlines() if p.strip()]
    clips = load_jsonl(args.manifest)[: args.limit]
    args.list_id = hashlib.md5("\n".join(phrases).encode()).hexdigest()[:8]  # rescore if the list changes
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    score_clips(args, phrases, clips, cache_path)

    from datamodule.transforms import TextTransform
    tt = TextTransform()
    n_tokens = np.array([len(tt.tokenize(p.upper())) for p in phrases])
    cached = {r["clip_id"]: r for r in load_jsonl(cache_path) if r["phrases"] == args.list_id}
    report(phrases, clips, cached, n_tokens)


if __name__ == "__main__":
    main()
