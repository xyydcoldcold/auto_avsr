# silent_speech: baseline test set and scoring

Kept in its own folder so the upstream auto_avsr files stay untouched. The only shared
code is `vsr_pipeline.py` at the repo root (`transcribe_nbest` was added for this).

Run everything from the repo root with the `auto_avsr` conda env.

```bash
# 1. Record the 80 prompts, silent and voiced (camera 1 on this Mac)
python silent_speech/record.py --session day1 --condition silent --camera 1
python silent_speech/record.py --session day1 --condition voiced --camera 1

# 2. Run the model over the manifest (top-10 n-best, beam 10, with latency)
python silent_speech/infer.py --model tutorials/vsr_trlrs2lrs3vox2avsp_base.pth

# 3. Score
python silent_speech/evaluate.py --pred silent_speech/outputs/predictions.jsonl \
    --manifest silent_speech/data/manifest.jsonl
```

- Clips go to `silent_speech/data/` (git-ignored, it is face video). They are saved at
  25 fps, resampled from the camera's timestamps.
- `infer.py` resumes: it only runs clips missing from the output file. Use a new `--out`
  per model or beam setting, e.g. `--out silent_speech/outputs/trlrs3_beam10.jsonl`.
- A clip with no detectable face is scored as an empty transcript (100% WER) and
  left out of the latency average, so failures are visible rather than skipped.
- These recordings are the fixed test set. Never train on them.
