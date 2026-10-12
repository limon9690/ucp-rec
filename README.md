# UCP-Rec

Reproduction of **UCP-Rec** (User-Conditioned Popularity Recommender) from my master's thesis
at Northeastern University (China), 2026. UCP-Rec extends
[PrepRec](https://github.com/CrowdDynamicsLab/preprec) (Wang, Rathi & Sundaram, RecSys 2024)
with a user popularity profile and three personalization modules (MoPE, FiLM, APF) for
zero-shot cross-domain sequential recommendation.

The original thesis code was lost, so this repo rebuilds it from the thesis on top of the
public PrepRec code. Progress and every deviation from the thesis are tracked in
[docs/REPRODUCTION_LOG.md](docs/REPRODUCTION_LOG.md).

**Status:** environment and PrepRec pipeline working; data and baseline runs next.

## Setup

### Local machine (editing and CPU debugging)

```bash
conda env create -f environment.yml
conda activate ucp-rec
bash scripts/smoke_test.sh cpu      # about a minute; must end with SMOKE TEST PASSED
```

The smoke test builds two small synthetic datasets, trains PrepRec for two epochs and runs
a zero-shot transfer. It checks the plumbing, not the results.

### GPU instance (AutoDL)

Image: **PyTorch 1.10.0 / Python 3.8 (ubuntu20.04) / CUDA 11.3**.

```bash
cd /root && git clone <this repo> ucp-rec && cd ucp-rec
bash scripts/setup_autodl.sh        # installs pins, links data/ and res/ to the data disk
bash scripts/smoke_test.sh cuda
```

Code lives on the system disk; datasets and checkpoints live on `/root/autodl-tmp`, which
survives shutdowns. Run long jobs inside `tmux`.

## Data

The five datasets are the filtered CSVs released by the PrepRec authors
([Google Drive folder](https://drive.google.com/drive/u/0/folders/1Jqyu615pUTnPkg0RFZ3MLQF7b8nCk-pE)).
Put them at:

```
data/amazon/amazon_office.csv   data/amazon/amazon_tool.csv
data/douban/douban_movie.csv    data/douban/douban_music.csv
data/epinions/epinions.csv
```

Then:

```bash
bash scripts/prepare_data.sh        # CPU only
python scripts/check_data.py --verify
```

`prepare_data.sh` checks user/item counts against thesis Table 3.3 and writes
`data/MANIFEST.md5`. That file is committed so anyone can confirm they evaluate on
byte-identical data, including the sampled negatives.

## Running PrepRec (baseline)

```bash
F="--time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj"
# train on Office
python main.py --dataset amazon/amazon_office --train_dir train $F
# zero-shot to Movie
python main.py --dataset douban/douban_movie --train_dir zs_from_office \
  --state_dict_path res/amazon/amazon_office/train/best.pth --transfer --inference_only $F
```

Each run prints and saves (`params.txt`) its trainable parameter count. PrepRec is 44,950.

## Changes to upstream PrepRec

Kept small and in separate commits so they are easy to audit:

- added `--pause` and `--fs_emb_2`, referenced by the code but missing from `parse.py`
  (without them `--use_week_eval` and zero-shot loading crash)
- evaluation negatives are seeded per user (`--seed` in `data_2.py`, default 2023);
  upstream drew them unseeded, so every preprocessing run gave a different test set
- preprocessing pools respect the container's CPU quota (`NUM_CPUS` overrides)
- `--n_workers` for the training sampler (default 4, same as upstream)
- trainable-parameter count printed and saved per run

## Credits

Built on [PrepRec](https://github.com/CrowdDynamicsLab/preprec), itself based on
[SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch). The PrepRec code here comes
from commit [`f7c3543`](https://github.com/CrowdDynamicsLab/preprec/tree/f7c354382151b4f3dc80905b55bc22c969184c1b);
its original README is in [docs/PREPREC_README.md](docs/PREPREC_README.md), and the
changes made to it are listed above.

```bibtex
@inproceedings{wang2024preprec,
  title     = {A Pre-trained Zero-shot Sequential Recommendation Framework via Popularity Dynamics},
  author    = {Wang, Junting and Rathi, Praneet and Sundaram, Hari},
  booktitle = {Proceedings of the 18th ACM Conference on Recommender Systems (RecSys)},
  year      = {2024}
}
```
