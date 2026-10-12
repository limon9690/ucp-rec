# UCP-Rec

UCP-Rec (User-Conditioned Popularity Recommender) is a zero-shot cross-domain sequential
recommender. Items are described only by how their popularity changes over time, so a
model trained on one domain can be applied to another without retraining.

This lets the model treat users who follow popular items differently from users with
niche taste. This is the code for UCP-Rec, proposed in my master's thesis (Northeastern
University, 2026).

## Setup

Requires Python 3.8 and PyTorch 1.10. then:

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

To run a full configuration (3 seeds, trained on Office, zero-shot on Tool, Movie,
Music and Epinions) and get mean ± sd:

```bash
bash scripts/run_variant.sh film   # or preprec, apf, mope, full, all
```

The summary is printed at the end and saved to `res/runs/<variant>/summary.txt`. Seeds
that already finished training are not retrained, only re-evaluated.

## Results

Trained on Amazon Office; Office is in-domain, the other four columns are zero-shot. NDCG@10 / HR@10, 100 sampled
negatives per test item.

| Model         | Params | Office        | Tool          | Movie         | Music         | Epinions      |
| ------------- | ------ | ------------- | ------------- | ------------- | ------------- | ------------- |
| PrepRec       | 44,950 | 0.336 / 0.532 | 0.308 / 0.473 | 0.613 / 0.814 | 0.474 / 0.679 | 0.429 / 0.629 |
| APF only      | 45,023 | 0.334 / 0.532 | 0.323 / 0.520 | 0.702 / 0.881 | 0.541 / 0.748 | 0.495 / 0.674 |
| FiLM only     | 48,614 | 0.336 / 0.535 | 0.330 / 0.531 | 0.698 / 0.881 | 0.526 / 0.738 | 0.494 / 0.664 |
| MoPE only     | 72,466 | 0.333 / 0.527 | 0.308 / 0.487 | 0.677 / 0.863 | 0.538 / 0.742 | 0.492 / 0.665 |
| UCP-Rec (all) | 76,203 | 0.336 / 0.534 | 0.326 / 0.521 | 0.682 / 0.867 | 0.539 / 0.743 | 0.491 / 0.670 |

## Notes

- The modules are switched on with `--apf`, `--film` and `--mope`; all three together is the
  full model. Their code is in `ucp.py`.

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
