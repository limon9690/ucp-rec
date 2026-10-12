#!/usr/bin/env bash
# End-to-end check on synthetic data: preprocess -> train 2 epochs -> zero-shot eval.
# Takes about a minute on a laptop CPU. Numbers it prints are meaningless;
# what matters is that it finishes and reports 44950 trainable parameters.
# Usage: bash scripts/smoke_test.sh [cpu|cuda]
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-cpu}"
COMMON="--time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj --device $DEVICE"

python3 scripts/make_toy_data.py --name toy --seed 0
python3 scripts/make_toy_data.py --name toy2 --seed 1 --items 450
(cd data && for d in toy toy2; do
  python3 data.py --dataset ./$d/$d --t1_cutoff 10 --t2_cutoff 2 >/dev/null
  python3 data_2.py --dataset ./$d/$d --userneg --week_adj >/dev/null
done)

rm -rf res/toy res/toy2
python3 main.py --dataset toy/toy --train_dir smoke_train --num_epochs 2 --epoch_test 1 $COMMON
python3 main.py --dataset toy2/toy2 --train_dir smoke_zs \
  --state_dict_path res/toy/toy/smoke_train/best.pth --transfer --inference_only $COMMON

params=$(cat res/toy/toy/smoke_train/params.txt)
if [[ "$params" == "44950" ]]; then
  echo "SMOKE TEST PASSED (44950 trainable parameters, matches thesis PrepRec)"
else
  echo "SMOKE TEST FAILED: $params trainable parameters, expected 44950"; exit 1
fi
