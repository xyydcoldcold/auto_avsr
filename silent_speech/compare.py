"""Compare two settings (A vs B) on the test set with paired bootstrap confidence intervals.

Usage (from the repo root):
    python silent_speech/compare.py \
        --a silent_speech/outputs/predictions_lora_300clips_avg_s0.jsonl silent_speech/outputs/predictions_lora_300clips_avg_s1.jsonl \
        --b silent_speech/outputs/predictions_lora_400clips_avg_s0.jsonl silent_speech/outputs/predictions_lora_400clips_avg_s1.jsonl \
        --name-a 300 --name-b 400

Each side takes one predictions file per training seed (infer.py output). Per clip, the errors are
averaged over that side's seeds, so the comparison is "mean of seeds A" vs "mean of seeds B".

Paired bootstrap: draw the test clips with replacement (10,000 times), recompute each metric for A and
B on the SAME drawn clips, and keep B - A. The middle 95% of those differences is the 95% CI. Pairing
matters: a clip that is hard for A is usually hard for B too, so the difference is far less noisy than
either number alone. "P(B better)" is the share of resamples where B wins.

What the CI does and does not cover: it covers test-set noise (which 100 / 40 sentences we happened
to record). It does not cover training noise (seed); that shows up in the per-seed columns. Trust a
difference only when the CI excludes 0 AND every seed of B beats every seed of A ("clear" below).

Metrics: WER (top-1), oWER@k (best of the top k hypotheses), sent@1 / sent@k (exact sentence match).
Groups: condition/split (silent/in_scene, silent/open, silent/out_scene, voiced/...), condition/ALL.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate import edit_distance, load_jsonl, norm  # noqa: E402

# name, numerator field, denominator field, lower is better
METRICS = [("WER", "w_err", "words", True), ("oWER@{k}", "w_err_k", "words", True),
           ("sent@1", "exact1", "one", False), ("sent@{k}", "exactk", "one", False)]
FIELDS = ["w_err", "w_err_k", "exact1", "exactk"]


def clip_stats(pred_file, ref, k):
    """{clip_id: {w_err, w_err_k, exact1, exactk}} for one predictions file."""
    out = {}
    for p in load_jsonl(pred_file):
        r = ref.get(p["clip_id"])
        if r is None or not p.get("nbest"):
            continue
        gold = norm(r["text"])
        hyps = [norm(h["text"]) for h in p["nbest"]]
        out[p["clip_id"]] = {
            "w_err": edit_distance(gold.split(), hyps[0].split()),
            "w_err_k": min(edit_distance(gold.split(), h.split()) for h in hyps[:k]),
            "exact1": float(hyps[0] == gold),
            "exactk": float(gold in hyps[:k]),
        }
    return out


def side_arrays(files, ref, k):
    """Per seed: {field: {clip_id: value}}; returns the list of per-seed dicts."""
    return [clip_stats(f, ref, k) for f in files]


def to_matrix(seeds, clip_ids, field):
    """seeds x clips array of one field."""
    return np.array([[s[c][field] for c in clip_ids] for s in seeds], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="Predictions of setting A, one file per seed")
    ap.add_argument("--b", nargs="+", required=True, help="Predictions of setting B, one file per seed")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--condition", default=None, help="Only this condition, e.g. silent")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    args = ap.parse_args()

    ref = {r["clip_id"]: r for r in load_jsonl(args.manifest)}
    seeds_a, seeds_b = side_arrays(args.a, ref, args.k), side_arrays(args.b, ref, args.k)
    common = set.intersection(*(set(s) for s in seeds_a + seeds_b))
    missing = max(len(s) for s in seeds_a + seeds_b) - len(common)
    if missing:
        print(f"note: {missing} clip(s) are missing from some files and are left out of every group")

    groups = {}
    for cid in sorted(common):
        r = ref[cid]
        if args.condition and r["condition"] != args.condition:
            continue
        for key in (f"{r['condition']}/{r['split']}", f"{r['condition']}/ALL"):
            groups.setdefault(key, []).append(cid)

    rng = np.random.default_rng(args.seed)
    na, nb = args.name_a, args.name_b
    print(f"A = {na}: {len(args.a)} seed(s)   B = {nb}: {len(args.b)} seed(s)   "
          f"paired bootstrap x{args.n_boot}, 95% CI of B - A")
    print("WER rows: negative B - A means B is better. sent rows: positive means B is better.")

    order = sorted(groups, key=lambda g: (not g.startswith("silent"), g.endswith("ALL"), g))
    for key in order:
        ids = groups[key]
        n = len(ids)
        words = np.array([len(norm(ref[c]["text"]).split()) for c in ids], dtype=float)
        one = np.ones(n)
        idx = rng.integers(0, n, size=(args.n_boot, n))  # same draws for every metric in this group
        print(f"\n{key}  (n={n})")
        print(f"  {'metric':8} {na + ' seeds':>16} {nb + ' seeds':>16} {na:>7} {nb:>7} {'B-A':>7} "
              f"{'95% CI':>17} {'P(B better)':>11}  verdict")
        for name, field, denom_name, lower_better in METRICS:
            denom = words if denom_name == "words" else one
            ma, mb = to_matrix(seeds_a, ids, field), to_matrix(seeds_b, ids, field)
            per_seed_a = ma.sum(1) / denom.sum()
            per_seed_b = mb.sum(1) / denom.sum()
            ca, cb = ma.mean(0), mb.mean(0)  # per clip, averaged over seeds
            val_a, val_b = ca.sum() / denom.sum(), cb.sum() / denom.sum()
            d_den = denom[idx].sum(1)
            diff = (cb[idx].sum(1) - ca[idx].sum(1)) / d_den
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p_better = float(np.mean(diff < 0) if lower_better else np.mean(diff > 0))
            ci_excludes_0 = hi < 0 or lo > 0
            b_wins = (diff.mean() < 0) == lower_better
            if lower_better:
                seeds_separate = per_seed_b.max() < per_seed_a.min() if b_wins else per_seed_a.max() < per_seed_b.min()
            else:
                seeds_separate = per_seed_b.min() > per_seed_a.max() if b_wins else per_seed_a.min() > per_seed_b.max()
            winner = nb if b_wins else na
            if ci_excludes_0 and seeds_separate:
                verdict = f"clear: {winner}"
            elif ci_excludes_0 or (seeds_separate and max(p_better, 1 - p_better) >= 0.9):
                verdict = f"leaning {winner}"
            else:
                verdict = "tie"
            fmt_seeds = lambda v: "/".join(f"{100 * x:.1f}" for x in v)  # noqa: E731
            label = name.format(k=args.k)
            print(f"  {label:8} {fmt_seeds(per_seed_a):>16} {fmt_seeds(per_seed_b):>16} "
                  f"{100 * val_a:>7.1f} {100 * val_b:>7.1f} {100 * (val_b - val_a):>+7.1f} "
                  f"{f'[{100 * lo:+.1f}, {100 * hi:+.1f}]':>17} {100 * p_better:>10.0f}%  {verdict}")


if __name__ == "__main__":
    main()
