# Usage: bash scripts/prepare_data.sh [dataset ...]   (default: all five)
set -euo pipefail
cd "$(dirname "$0")/../data"

ALL="amazon/amazon_office amazon/amazon_tool douban/douban_movie douban/douban_music epinions/epinions"
DATASETS="${*:-$ALL}"
SEED="${NEG_SEED:-2023}"

for ds in $DATASETS; do
  [[ -f "$ds.csv" ]] || { echo "missing data/$ds.csv"; exit 1; }
  echo "== $ds"
  if [[ "$ds" == "amazon/amazon_tool" ]]; then
    python3 data.py --dataset "./$ds"
  else
    python3 data.py --dataset "./$ds" --t1_cutoff 10 --t2_cutoff 2
  fi
  python3 data_2.py --dataset "./$ds" --userneg --week_adj --seed "$SEED"
done

cd ..
python3 scripts/check_data.py $DATASETS
