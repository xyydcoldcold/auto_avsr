"""Closed phrase set check: pick the answer from a fixed phrase list instead of decoding freely.

Usage (from the repo root):
    python silent_speech/closed_set.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth

For every clip, the model scores every phrase in phrases.txt (teacher forcing: decoder + CTC,
weighted 0.9 / 0.1 like the beam search). Scores are cached in outputs/phrase_scores.jsonl,
so re-runs only score new clips. Then, for each list size N, it reports:

    top1 / top3   in-scene clips: the right phrase is ranked 1st / in the top 3
    ms            time to score N phrases for one clip (encoding not included), scaled from the
                  timed full list since scoring time grows linearly with the number of phrases

and a rejection table (silent only; voiced is near perfect) comparing confidence rules:

    margin        best phrase score minus 2nd best (phrases only compared with each other)
    beam gap      best phrase score minus the free beam-search top score (predictions.jsonl),
                  per encoder frame: how much worse the list fits than the model's own best guess
    ctc gap       same idea with the CTC head only: best phrase CTC score minus the greedy CTC
                  path (argmax per frame), per frame. Nearly free at run time, unlike beam search.

    FA@FRx        open + out-of-scene clips (not in the list) accepted as some phrase, with the
                  reject threshold set so x of in-scene clips are rejected
    AUC           chance a random in-scene clip gets higher confidence than a random non-list clip
    acc@FR20      top1 accuracy among the in-scene clips that are accepted at FR 20%

The held-out table fits the FR 20% threshold on --fit-session (day1) and applies it unchanged to
--test-session (day2): FR and FA there are what a fixed threshold would really do on a new day.

Lists of size N always contain the 50 in-scene test prompts plus N-50 random other phrases,
averaged over several random draws. Thresholds are set on the same clips they are scored on,
so FA is optimistic until it is checked on another session.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# The visual frontend's 3D max-pool has no MPS kernel; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from infer import load_jsonl, read_frames  # noqa: E402

CTC_WEIGHT = 0.1
SIZES = [50, 100, 250, 500]
SEEDS = 5
FALSE_REJECTS = [0.1, 0.2, 0.3]
RULES = ["margin", "beam gap", "ctc gap"]


def norm(s):
    return " ".join(s.lower().split())


def ctc_greedy(pipeline, enc):
    """Log-prob of the single best CTC path (argmax token per frame) and the number of frames."""
    logp = pipeline.modelmodule.model.ctc.log_softmax(enc)[0]  # T x V
    return float(logp.max(-1).values.sum()), enc.size(1)


def score_clips(args, phrases, clips, cache_path, greedy_path):
    from vsr_pipeline import VSRPipeline

    done = {r["clip_id"] for r in load_jsonl(cache_path) if r["phrases"] == args.list_id and r.get("device", "cpu") == args.device}
    have_greedy = {r["clip_id"] for r in load_jsonl(greedy_path)}
    pending = [c for c in clips if c["clip_id"] not in done or c["clip_id"] not in have_greedy]
    print(f"{len(clips) - len(pending)} clips already scored, scoring {len(pending)}")
    if not pending:
        return

    pipeline = VSRPipeline(args.model, device=args.device)
    tokens = [pipeline.tokenize(p) for p in phrases]
    with open(cache_path, "a", encoding="utf-8") as f, open(greedy_path, "a", encoding="utf-8") as g:
        for n, clip in enumerate(pending, 1):
            enc = pipeline.encode(read_frames(Path(args.manifest).parent / clip["path"]))
            if clip["clip_id"] not in have_greedy:
                greedy, frames = ctc_greedy(pipeline, enc)
                g.write(json.dumps({"clip_id": clip["clip_id"], "ctc_greedy": greedy, "frames": frames}) + "\n")
                g.flush()
            if clip["clip_id"] in done:
                print(f"[{n}/{len(pending)}] {clip['clip_id']}  ctc greedy only")
                continue
            start = time.perf_counter()
            att, ctc = pipeline.score_phrases(enc, tokens)
            ms = round((time.perf_counter() - start) * 1000)
            f.write(json.dumps({"clip_id": clip["clip_id"], "phrases": args.list_id, "device": args.device, "att": att,
                                "ctc": [max(c, -1e9) for c in ctc], "ms": ms}) + "\n")
            f.flush()
            print(f"[{n}/{len(pending)}] {clip['clip_id']}  {ms} ms")


def scores(row, n_tokens, length_norm):
    s = (1 - CTC_WEIGHT) * np.array(row["att"]) + CTC_WEIGHT * np.array(row["ctc"])
    return s / (n_tokens + 1) if length_norm else s  # +1 for <eos>


warnings.filterwarnings("ignore", message="Mean of empty slice")  # e.g. every test clip rejected


def auc(pos, neg):
    pos, neg = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def report(phrases, clips, cached, n_tokens, beam_top, greedy, fit_session=None, test_session=None):
    index = {p: i for i, p in enumerate(phrases)}
    test_ids = sorted({index[norm(c["text"])] for c in clips if c["split"] == "in_scene"})
    others = [i for i in range(len(phrases)) if i not in set(test_ids)]
    sessions = sorted({c["session"] for c in clips})

    accuracy, rejection, heldout = [], [], []
    for condition in ("silent", "voiced"):
        group = [c for c in clips if c["condition"] == condition and c["clip_id"] in cached]
        if not any(c["split"] == "in_scene" for c in group):
            continue
        ms = np.mean([cached[c["clip_id"]]["ms"] for c in group])
        for length_norm in (False, True):
            label = "per-token" if length_norm else "sum"
            for size in SIZES:
                if size > len(phrases):
                    continue
                seeds = 1 if size == len(test_ids) else SEEDS
                # per session: top1, top3, and per rule FA at each FR, AUC, acc@FR20 (thresholds in-sample)
                acc = {ses: {"top1": [], "top3": []} for ses in sessions}
                rej = {ses: {r: {"fa": [[] for _ in FALSE_REJECTS], "auc": [], "acc": []} for r in RULES}
                       for ses in sessions}
                held = {r: {"fr": [], "fa": [], "acc": []} for r in RULES}
                for seed in range(seeds):
                    subset = np.array(test_ids + random.Random(seed).sample(others, size - len(test_ids)))

                    def ranked(clip):
                        row = cached[clip["clip_id"]]
                        s = scores(row, n_tokens, length_norm)[subset]
                        order = subset[np.argsort(-s)]
                        best = order[0]
                        g = greedy.get(clip["clip_id"])
                        frames = g["frames"] if g else clip.get("n_frames", 1)
                        total = scores(row, n_tokens, False)  # gaps compare sums, the beam score's scale
                        conf = {
                            "margin": np.sort(s)[-1] - np.sort(s)[-2],
                            "beam gap": (total[best] - beam_top[clip["clip_id"]]) / frames
                            if clip["clip_id"] in beam_top else np.nan,
                            "ctc gap": (row["ctc"][best] - g["ctc_greedy"]) / frames if g else np.nan,
                        }
                        return order, conf

                    split = {}  # session -> (correct, top3 hits, {rule: pos conf}, {rule: neg conf})
                    for ses in sessions:
                        pos = [c for c in group if c["session"] == ses and c["split"] == "in_scene"]
                        neg = [c for c in group if c["session"] == ses and c["split"] != "in_scene"]
                        if not pos:
                            continue
                        pos_r, neg_r = [ranked(c) for c in pos], [ranked(c) for c in neg]
                        gold = [index[norm(c["text"])] for c in pos]
                        correct = np.array([r[0] == y for (r, _), y in zip(pos_r, gold)])
                        acc[ses]["top1"].append(correct.mean())
                        acc[ses]["top3"].append(np.mean([y in r[:3] for (r, _), y in zip(pos_r, gold)]))
                        pc = {r: np.array([c[r] for _, c in pos_r]) for r in RULES}
                        nc = {r: np.array([c[r] for _, c in neg_r]) for r in RULES}
                        split[ses] = (correct, pc, nc)
                        for rule in RULES:
                            if np.isnan(pc[rule]).any() or np.isnan(nc[rule]).any() or not len(nc[rule]):
                                continue
                            for k, fr in enumerate(FALSE_REJECTS):
                                thr = np.quantile(pc[rule], fr)
                                rej[ses][rule]["fa"][k].append(np.mean(nc[rule] >= thr))
                                if fr == 0.2:
                                    rej[ses][rule]["acc"].append(correct[pc[rule] >= thr].mean())
                            rej[ses][rule]["auc"].append(auc(pc[rule], nc[rule]))

                    # held out: threshold for FR 20% on the fit session, applied as-is to the test session
                    if fit_session in split and test_session in split:
                        _, fit_pc, _ = split[fit_session]
                        correct, pc, nc = split[test_session]
                        for rule in RULES:
                            if np.isnan(fit_pc[rule]).any() or np.isnan(pc[rule]).any() or np.isnan(nc[rule]).any():
                                continue
                            thr = np.quantile(fit_pc[rule], 0.2)
                            held[rule]["fr"].append(np.mean(pc[rule] < thr))
                            held[rule]["fa"].append(np.mean(nc[rule] >= thr))
                            accepted = pc[rule] >= thr
                            held[rule]["acc"].append(correct[accepted].mean() if accepted.any() else np.nan)

                for ses in sessions:
                    if not acc[ses]["top1"]:
                        continue
                    accuracy.append(f"{condition:8} {ses:7} {label:10} {size:>4} {np.mean(acc[ses]['top1']):>6.1%} "
                                    f"{np.mean(acc[ses]['top3']):>6.1%} {ms * size / len(phrases):>6.0f}")
                    if condition != "silent":
                        continue
                    for rule in RULES:
                        st = rej[ses][rule]
                        if not st["auc"]:
                            rejection.append(f"{ses:7} {label:10} {size:>4} {rule:9}  (missing data)")
                            continue
                        fas = " ".join(f"{np.mean(f):>8.1%}" for f in st["fa"])
                        rejection.append(f"{ses:7} {label:10} {size:>4} {rule:9} {fas} {np.mean(st['auc']):>6.2f} "
                                         f"{np.mean(st['acc']):>9.1%}")
                if condition == "silent" and fit_session and test_session:
                    for rule in RULES:
                        h = held[rule]
                        if h["fr"]:
                            heldout.append(f"{label:10} {size:>4} {rule:9} {np.mean(h['fr']):>7.1%} "
                                           f"{np.mean(h['fa']):>7.1%} {np.nanmean(h['acc']):>9.1%}")

    print(f"\n{'cond':8} {'session':7} {'score':10} {'N':>4} {'top1':>6} {'top3':>6} {'ms':>6}")
    print("\n".join(accuracy))
    fa_head = " ".join(f"{'FA@FR' + str(int(fr * 100)):>8}" for fr in FALSE_REJECTS)
    print(f"\nRejection, silent, threshold fit on the same session\n"
          f"{'session':7} {'score':10} {'N':>4} {'rule':9} {fa_head} {'AUC':>6} {'acc@FR20':>9}")
    print("\n".join(rejection))
    if heldout:
        print(f"\nRejection, silent, threshold for FR 20% fit on {fit_session}, applied to {test_session}\n"
              f"{'score':10} {'N':>4} {'rule':9} {'FR':>7} {'FA':>7} {'acc':>9}")
        print("\n".join(heldout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a VSR .pth checkpoint")
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.jsonl"))
    ap.add_argument("--phrases", default=str(HERE / "phrases.txt"))
    ap.add_argument("--cache", default=str(HERE / "outputs" / "phrase_scores.jsonl"))
    ap.add_argument("--pred", default=str(HERE / "outputs" / "predictions.jsonl"),
                    help="infer.py output; its top beam score is the reference for the beam gap")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="mps = Apple GPU")
    ap.add_argument("--fit-session", default="day1", help="session the reject threshold is fit on")
    ap.add_argument("--test-session", default="day2", help="session the fitted threshold is checked on")
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N clips (for testing)")
    args = ap.parse_args()

    phrases = [norm(p) for p in Path(args.phrases).read_text(encoding="utf-8").splitlines() if p.strip()]
    clips = load_jsonl(args.manifest)[: args.limit]
    args.list_id = hashlib.md5("\n".join(phrases).encode()).hexdigest()[:8]  # rescore if the list changes
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    greedy_path = cache_path.with_name("ctc_greedy.jsonl")
    score_clips(args, phrases, clips, cache_path, greedy_path)

    from datamodule.transforms import TextTransform
    tt = TextTransform()
    n_tokens = np.array([len(tt.tokenize(p.upper())) for p in phrases])
    cached = {r["clip_id"]: r for r in load_jsonl(cache_path)
              if r["phrases"] == args.list_id and r.get("device", "cpu") == args.device}
    beam_top = {p["clip_id"]: p["nbest"][0]["score"] for p in load_jsonl(args.pred)
                if p["nbest"] and p["nbest"][0]["score"] is not None}
    greedy = {r["clip_id"]: r for r in load_jsonl(greedy_path)}
    report(phrases, clips, cached, n_tokens, beam_top, greedy, args.fit_session, args.test_session)


if __name__ == "__main__":
    main()
