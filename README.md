# UCP-Rec

UCP-Rec (User-Conditioned Popularity Recommender) is a zero-shot cross-domain sequential
recommender. It extends [PrepRec](https://github.com/CrowdDynamicsLab/preprec) with a
user popularity profile and three personalization modules (MoPE, FiLM and APF), so the
model can adapt to each user's taste for popular or niche items without seeing the
target domain during training.

## Setup

Python 3.8 and PyTorch 1.10. Install the PyTorch build for your machine from
[pytorch.org](https://pytorch.org/get-started/previous-versions/#v1101), then:

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh cpu      # or cuda; ends with SMOKE TEST PASSED
```

The smoke test runs the full pipeline (preprocessing, training, zero-shot transfer) on
small synthetic datasets in about a minute.

## Data

The experiments use the five datasets released by the PrepRec authors
([Google Drive](https://drive.google.com/drive/u/0/folders/1Jqyu615pUTnPkg0RFZ3MLQF7b8nCk-pE)):
Amazon Office and Tool, Douban Movie and Music, and Epinions. Place the CSVs at:

```
data/amazon/amazon_office.csv   data/amazon/amazon_tool.csv
data/douban/douban_movie.csv    data/douban/douban_music.csv
data/epinions/epinions.csv
```

Then preprocess (CPU only):

```bash
bash scripts/prepare_data.sh
python scripts/check_data.py --verify   # confirms byte-identical evaluation data
```

## Usage

Train on Office, then evaluate zero-shot on Movie:

```bash
F="--time_embed --monthpop wtembed --weekpop week_embed2 --use_week_eval --week_eval_pop week_wt_embed_adj"

python main.py --dataset amazon/amazon_office --train_dir train $F

python main.py --dataset douban/douban_movie --train_dir zs_from_office \
  --state_dict_path res/amazon/amazon_office/train/best.pth --transfer --inference_only $F
```

Results and checkpoints are written to `res/<dataset>/<train_dir>/`.

To run a full configuration (seeds 2023–2025, trained on Office, zero-shot on Tool, Movie,
Music and Epinions) and get mean ± sd:

```bash
nohup bash scripts/run_variant.sh film > film.out 2>&1 &   # or preprec, apf
cat res/runs/film/summary.txt                              # when finished
```

Seeds that already finished training are not retrained, only re-evaluated.

## Acknowledgements

Built on [PrepRec](https://github.com/CrowdDynamicsLab/preprec) by Junting Wang,
Praneet Rathi and Hari Sundaram, which is in turn based on
[SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch). If you use this code, please
cite their work:

```bibtex
@inproceedings{wang2024preprec,
  title     = {A Pre-trained Zero-shot Sequential Recommendation Framework via Popularity Dynamics},
  author    = {Wang, Junting and Rathi, Praneet and Sundaram, Hari},
  booktitle = {Proceedings of the 18th ACM Conference on Recommender Systems (RecSys)},
  year      = {2024}
}
```
