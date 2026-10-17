# Silent speech: personalized VSR on Auto-AVSR

Recognizing one person's **silent speech** (words mouthed with no voice) from a webcam, starting from
the pretrained Auto-AVSR visual speech recognition model.

## Motivation

Public lip-reading models (LRS2, LRS3, VoxCeleb2, AVSpeech) are trained on people *speaking aloud*,
with the audio removed. Silent articulation is a different input: smaller mouth movements, no voicing
cues in the face, different timing. That this hurts accuracy is expected; the goal here is to
**quantify the gap on one user, and measure how much personal silent data closes it**, then build the
pieces a usable silent-input tool needs on top (n-best output, closed phrase sets, context reranking,
corrections).

## Pipeline

Everything lives in this folder so the upstream Auto-AVSR files stay untouched. The only shared code is
`vsr_pipeline.py` at the repo root (face crop with mediapipe, `transcribe_nbest()`, plus `encode()`,
`tokenize()`, `score_phrases()` for the closed-set check).

| script | what it does |
|---|---|
| `record.py` | Webcam recorder: shows each prompt, saves one clip per prompt to `data/` at 25 fps (resampled from camera timestamps), appends to `data/manifest.jsonl`. |
| `infer.py` | Runs the VSR model over the manifest with beam search; keeps the top-10 hypotheses with scores and per-clip latency in `outputs/predictions.jsonl`. Resumes: only runs clips not yet in the output file. |
| `evaluate.py` | Scores predictions: WER, CER, oracle WER@k, sentence accuracy @1/@k, latency, grouped by condition x split x session. No third-party dependencies. |
| `make_phrases.py` | Builds `phrases.txt`, the closed phrase pool (570 short chat messages, including deliberate look-alikes such as "see you at five" / "see you at nine"). |
| `closed_set.py` | Closed-set check: instead of decoding freely, the model scores every phrase against the clip (teacher-forced decoder + CTC, 0.9 / 0.1, same weights as beam search) and ranks them. Scores are cached in `outputs/phrase_scores.jsonl`. |

## How to run

Run from the repo root in the `auto_avsr` conda env. Always use `python`, not `python3` (on this Mac
`python3` points at a system Python without the dependencies).

```bash
conda activate auto_avsr

# 1. Record the 80 prompts, silent and voiced (camera 1 on this Mac; 0 is Continuity Camera)
python silent_speech/record.py --session day1 --condition silent --camera 1
python silent_speech/record.py --session day1 --condition voiced --camera 1

# 2. Run the model over the manifest (beam 10, top-10 kept, with latency)
python silent_speech/infer.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth

# 3. Score free decoding
python silent_speech/evaluate.py --pred silent_speech/outputs/predictions.jsonl \
    --manifest silent_speech/data/manifest.jsonl --k 5

# 4. Closed phrase set check
python silent_speech/make_phrases.py
python silent_speech/closed_set.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth
```

Notes:
- Clips go to `silent_speech/data/` and results to `silent_speech/outputs/`; both are git-ignored
  (face video, per-run tables).
- Use a new `--out` per model or beam setting, e.g. `--out silent_speech/outputs/trlrs3_beam10.jsonl`.
- To re-record a clip, delete its rows from `data/manifest.jsonl` and `outputs/predictions.jsonl`,
  then rerun `record.py`.
- A clip with no detectable face is scored as an empty transcript (100% WER) and left out of the
  latency average, so failures stay visible.
- Add `--limit N` to `infer.py` or `closed_set.py` for a quick check on the first N clips.

## Benchmark design

- **80 prompts** (`prompts.csv`), short (~4 words on average) so splits differ by content, not length:
  - 50 **in-context**: everyday short chat messages ("on my way", "running ten minutes late"),
  - 20 **open**: general sentences outside that scene,
  - 10 **out-of-context**: unusual content, used to check over-biasing toward the scene.
- **Silent vs voiced**: the same 80 prompts recorded both ways in the same session, so the gap is
  measured on identical sentences, face and lighting.
- **Multiple sessions**: the same prompts are recorded on different days (`--session day1`, `day2`, ...)
  to measure cross-day stability.
- **Fixed test set**: these recordings are never used for training, and the prompt list is frozen once
  a round has recorded it. Personal training data is recorded separately.
- Metrics: WER, CER, oracle WER@k (best of the top k), sentence accuracy @1 / @k, latency per clip.
  For the closed set: top-1 / top-3 accuracy on in-context clips, false accepts of open and
  out-of-context clips at a 20% false-reject rate, and scoring time.

## Results so far

Model `vsr_trlrs2lrs3vox2avsp_base.pth` (20.3% WER on LRS3), beam 10, M2 Pro CPU, day 1 only.

**Free decoding** (k = 5):

| group | n | WER | CER | oWER@5 | sent@1 | sent@5 |
|---|---:|---:|---:|---:|---:|---:|
| silent, all | 80 | 73.6% | 50.2% | 63.5% | 20.0% | 23.8% |
| silent, in-context | 50 | 69.3% | 53.2% | 59.0% | 24.0% | 28.0% |
| silent, open | 20 | 62.2% | 35.6% | 52.4% | 20.0% | 25.0% |
| silent, out-of-context | 10 | 116.7% | 65.3% | 107.1% | 0.0% | 0.0% |
| voiced, all | 80 | 31.0% | 19.3% | 20.4% | 50.0% | 63.7% |
| voiced, in-context | 50 | 20.5% | 14.9% | 9.8% | 62.0% | 76.0% |

- Silent is far worse than voiced on the same sentences (in-context WER 69% vs 20%), so the personal
  training set has to be recorded silent.
- For silent clips the right sentence is rarely in the top 5 either (sent@5 28%), so reranking alone
  can't fix it: the visual side comes first. Voiced shows reranking does help once the visual side is
  good (sent 62% to 76%).
- Many silent errors are viseme near misses ("what time works for you" decoded as "one time weeks for you").
- Out-of-context has only 10 clips and WER over 100% from insertions; treat it as noisy.
- Latency about 1.8 s per clip (crop ~0.4 s, decode ~1.45 s): fine for push-to-talk, not real time.

**Closed phrase set** (model scores each phrase; lists of size N always contain the 50 in-context
prompts plus random others from the pool, averaged over 5 draws):

| condition | score | N | top-1 | top-3 | false accept @ 20% false reject | ms / clip |
|---|---|---:|---:|---:|---:|---:|
| silent | sum | 50 | 84.0% | 92.0% | 13.3% | 135 |
| silent | sum | 500 | 58.0% | 68.4% | 66.0% | 1353 |
| silent | length-normalized | 50 | 76.0% | 96.0% | 96.7% | 135 |
| silent | length-normalized | 500 | 66.4% | 80.8% | 75.3% | 1353 |
| voiced | either | 50 to 500 | 96 to 100% | 98 to 100% | 0 to 7% | 148 to 1478 |

- Picking 1 of N phrases works much better than spelling the sentence: silent top-1 84% at N = 50
  vs 24% exact sentences from free decoding. With 500 phrases, length-normalized scoring keeps
  66% top-1 / 81% top-3.
- Rejecting sentences that aren't in the list does not work yet: the top-1 vs top-2 margin rule lets
  through 56 to 97% of them, except sum scoring at N = 50 (13%).
- Scoring costs about 2.8 ms per phrase on CPU, so lists above ~100 phrases exceed a 300 ms budget.
- Caveats: the reject threshold is tuned on the same clips (optimistic), and with 50 positive clips
  differences under ~10 points are noise. Day 2 will check both.

Still to measure: day 2 silent (cross-day) [TBD], closed set on day 2 [TBD], LoRA results [TBD].

## Roadmap

1. **Day 2 silent recording**: cross-day stability, and a held-out check of the closed-set threshold.
2. **Better reject rule**: compare the best phrase score with the free-decoding top score; then
   **MPS latency** for phrase scoring.
3. **Personal LoRA**: freeze the backbone, train a small LoRA/adapter on a separate set of personal
   silent recordings (GPU). Report WER vs minutes of calibration data
   (e.g. 0 / 1 / 5 / 10 / 30 min) [TBD].
4. **Context reranking**: rescore the n-best with scene vocabulary, preceding text, and a small LM;
   measure in-context gain against over-biasing of out-of-context sentences.
5. **Correction loop**: user picks and edits update the phrase list and biasing weights right away;
   later, periodically fold reliable corrections back into the adapter.
6. **Pseudo-streaming**: detect when the lips stop moving to end a clip automatically.
   True streaming (e.g. Emformer) is deferred.
