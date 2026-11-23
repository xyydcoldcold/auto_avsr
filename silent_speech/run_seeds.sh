#!/usr/bin/env bash
# Train one LoRA setting with several seeds and run inference for each, so compare.py can judge
# the setting on the mean of seeds (one seed alone moves in-scene WER by ~8 points).
#
# Usage (from the repo root, after `conda activate auto_avsr`):
#   bash silent_speech/run_seeds.sh 300            # seeds 0 and 1
#   bash silent_speech/run_seeds.sh 400 "0 1 2"    # three seeds
#   EXTRA="--epochs 25" TAG=ep25 bash silent_speech/run_seeds.sh 300   # a variant with its own names
#
# Writes adapters/lora_<N>clips[_<TAG>]_avg_s<seed>.pt and
# outputs/predictions_lora_<N>clips[_<TAG>]_avg_s<seed>.jsonl. Already-finished seeds are skipped
# (training if the adapter exists, inference resumes clip by clip). Don't run while recording.
set -euo pipefail
N=${1:?usage: run_seeds.sh <clips> ["seeds"]}
SEEDS=${2:-"0 1"}
MODEL=${MODEL:-tutorials/vsr_trlrs2lrs3vox2avsp_base.pth}
DEVICE=${DEVICE:-mps}
EXTRA=${EXTRA:-}
NAME="lora_${N}clips${TAG:+_$TAG}"

for s in $SEEDS; do
  adapter="silent_speech/adapters/${NAME}_avg_s${s}.pt"
  if [ -f "$adapter" ]; then
    echo "== seed $s: $adapter exists, skip training"
  else
    echo "== seed $s: training"
    # shellcheck disable=SC2086
    python -u silent_speech/train_lora.py --model "$MODEL" --device "$DEVICE" --clips "$N" --seed "$s" \
      --out "$adapter" $EXTRA
  fi
  echo "== seed $s: inference"
  python -u silent_speech/infer.py --model "$MODEL" --adapter "$adapter"
done

echo
echo "Done. Compare with e.g.:"
echo "  python silent_speech/compare.py --a silent_speech/outputs/predictions_<A>_avg_s{0,1}.jsonl \\"
echo "      --b silent_speech/outputs/predictions_${NAME}_avg_s{0,1}.jsonl"
