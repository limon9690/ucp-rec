#!/bin/bash
# Train one configuration on the source dataset for each seed, evaluate zero-shot
# on the target datasets, and print mean ± sd.
#
#   nohup bash scripts/run_variant.sh preprec|apf|film
#
# Training is skipped for a seed whose run already finished, so rerunning only
# fills in what is missing. Results go to res/runs/<variant>/.
set -euo pipefail
cd "$(dirname "$0")/.."

variant=${1:?usage: bash scripts/run_variant.sh preprec|apf|film}
case $variant in
  preprec) flags="" ;;
  apf) flags="--apf" ;;
  film) flags="--film" ;;
  *) echo "unknown variant: $variant"; exit 1 ;;
esac

SEEDS=${SEEDS:-"2023 2024 2025"}
SOURCE=${SOURCE:-amazon/amazon_office}
TARGETS=${TARGETS:-"amazon/amazon_tool douban/douban_movie douban/douban_music epinions/epinions"}
EXTRA=${EXTRA:-}
F="--time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj"

out=res/runs/$variant
mkdir -p "$out"
results=$out/results.txt
: > "$results"

for s in $SEEDS; do
  run=${variant}_s$s
  log=res/$SOURCE/$run/log.txt
  if [[ -f $log ]] && grep -q "Test results" "$log"; then
    echo "$run: already trained, skipping training"
  else
    echo "$run: training on $SOURCE"
    python -u main.py --dataset "$SOURCE" --train_dir "$run" --seed "$s" $flags $F $EXTRA > "$out/train_s$s.log" 2>&1
  fi
  line=$(grep -A1 "Test results" "$log" | tail -1)
  ndcg=$(echo "$line" | sed -E 's/.*NDCG@10: ([0-9.]+).*/\1/')
  hr=$(echo "$line" | sed -E 's/.*HR@10: ([0-9.]+).*/\1/')
  echo "$SOURCE(in-domain) $s $ndcg $hr" >> "$results"

  for d in $TARGETS; do
    echo "$run: zero-shot on $d"
    res=$(python -u main.py --dataset "$d" --train_dir "zs_$run" --state_dict_path "res/$SOURCE/$run/best.pth" \
          --transfer --inference_only $flags $F $EXTRA 2>&1 | tee "$out/zs_${d//\//_}_s$s.log" | grep "^\[\[" | tail -1)
    echo "$d $s $(echo "$res" | sed -E 's/^\[\[([0-9.]+), ([0-9.]+)\].*/\1 \2/')" >> "$results"
  done
done

python scripts/summarize.py "$results" | tee "$out/summary.txt"
