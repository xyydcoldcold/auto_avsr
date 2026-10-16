"""Score model predictions against the manifest.

Usage:
    python evaluate.py --pred predictions.jsonl [--manifest data/manifest.jsonl]

predictions.jsonl, one line per clip:
    {"clip_id": "...", "nbest": [{"text": "...", "score": -3.2}, ...],
     "latency_ms": 840, "model": "vsr_trlrs2lrs3vox2avsp_base", "adapter": null}

Prints WER / CER / oracle top-k WER / latency, grouped by condition x split x session.
No third-party dependencies.
"""
import argparse
import json
import re
from collections import defaultdict


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    ref = {r["clip_id"]: r for r in load_jsonl(args.manifest)}
    groups = defaultdict(lambda: defaultdict(float))

    for p in load_jsonl(args.pred):
        r = ref.get(p["clip_id"])
        if r is None or not p.get("nbest"):
            continue
        gold_w, gold_c = norm(r["text"]).split(), norm(r["text"])
        hyps = [norm(h["text"]) for h in p["nbest"]]
        w1 = edit_distance(gold_w, hyps[0].split())
        c1 = edit_distance(list(gold_c), list(hyps[0]))
        wk = min(edit_distance(gold_w, h.split()) for h in hyps[: args.k])
        exact_k = any(h == gold_c for h in hyps[: args.k])
        for key in ("ALL", f"{r['condition']}/{r['split']}", f"{r['condition']}/session={r['session']}"):
            g = groups[key]
            g["n"] += 1
            g["words"] += len(gold_w)
            g["chars"] += len(gold_c)
            g["w_err"] += w1
            g["c_err"] += c1
            g["w_err_k"] += wk
            g["exact1"] += hyps[0] == gold_c
            g["exactk"] += exact_k
            if "error" not in p:  # failed clips (no face found) have no real latency
                g["lat"] += p.get("latency_ms", 0)
                g["lat_n"] += 1

    print(f"{'group':32} {'n':>4} {'WER':>7} {'CER':>7} {f'oWER@{args.k}':>8} "
          f"{'sent@1':>7} {f'sent@{args.k}':>7} {'lat ms':>7}")
    for key in sorted(groups, key=lambda k: (k != "ALL", k)):
        g = groups[key]
        print(f"{key:32} {int(g['n']):>4} "
              f"{g['w_err'] / g['words']:>7.1%} {g['c_err'] / g['chars']:>7.1%} "
              f"{g['w_err_k'] / g['words']:>8.1%} "
              f"{g['exact1'] / g['n']:>7.1%} {g['exactk'] / g['n']:>7.1%} "
              f"{g['lat'] / max(g['lat_n'], 1):>7.0f}")


if __name__ == "__main__":
    main()
