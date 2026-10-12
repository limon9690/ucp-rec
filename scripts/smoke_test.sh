
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
check() {
  local name=$1 flags=$2 expected=$3
  python3 main.py --dataset toy/toy --train_dir smoke_$name --num_epochs 2 --epoch_test 1 $COMMON $flags
  python3 main.py --dataset toy2/toy2 --train_dir smoke_${name}_zs \
    --state_dict_path res/toy/toy/smoke_$name/best.pth --transfer --inference_only $COMMON $flags
  local params=$(cat res/toy/toy/smoke_$name/params.txt)
  if [[ "$params" != "$expected" ]]; then
    echo "SMOKE TEST FAILED: $name has $params trainable parameters, expected $expected"; exit 1
  fi
}

check preprec "" 44950
check apf "--apf" 45023
echo "SMOKE TEST PASSED (PrepRec 44950, APF 45023 parameters)"
