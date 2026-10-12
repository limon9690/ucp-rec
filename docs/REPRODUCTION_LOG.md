# Reproduction log

Where the thesis is silent or the upstream code disagrees with it, the decision and the
reason go here. Thesis section and table numbers refer to the 2026 NEU master's thesis.

## Targets

| Check | Thesis | Reproduced |
| --- | --- | --- |
| PrepRec trainable parameters | 44,950 | 44,950 (toy data) |
| PrepRec Office NDCG@10 (in-domain) | 0.336 | |
| PrepRec Office→Movie | 0.613 | |
| PrepRec Office→Music | 0.474 | |
| PrepRec Office→Epinions | 0.429 | |
| UCP-Rec trainable parameters | 76,203 | |
| UCP-Rec Office→Movie | 0.682 | |

Dataset sizes (Table 3.3) are checked by `scripts/check_data.py`.

## Findings so far

**Settled by the upstream defaults (`parse.py`):** dropout 0.2, max 80 epochs,
validation every 4 epochs, early stop after 3, lr 0.001, weight decay 1e-5,
batch 128, max length 200, hidden 50, 2 blocks, 1 head, seed 2023.

**Evaluation uses 100 negatives, not 99.** `data_2.py` samples 100 negatives per user
and evaluation ranks 101 candidates. The thesis text says 99. Since the thesis was built
on this code, the thesis wording is most likely the error; numbers here use 100.

**Upstream evaluation negatives were not reproducible.** They were drawn with the
unseeded global RNG inside forked pool workers, so every preprocessing run produced a
different test set. Now seeded per user. This means the exact negatives behind the
thesis numbers cannot be recovered, which is one reason to expect small differences.

**Training is not bit-reproducible with the default 4 sampler workers.** Workers push
batches into a shared queue, so batch order varies between runs with the same seed.
On toy data two identical runs differed by ~0.003 NDCG@5; with `--n_workers 1` they were
identical. Kept the upstream default of 4 for speed and comparability; run-to-run
variance is handled by reporting several seeds.

**Upstream crashed on its own sample commands** (`--use_week_eval`, zero-shot loading)
because `--pause` and `--fs_emb_2` were never defined. Fixed.

## Open decisions

- [ ] Train UCP-Rec from scratch or warm-start from the PrepRec checkpoint (Section 3.3.7
      is unclear). Try from scratch first.
- [ ] Formula for the fine-grained percentile of the 6 weekly profile features.
- [ ] Group-binning rule: fixed 0.2-wide bins (Ch. 3) vs. the observed range split into
      five (Ch. 4). Pick the one that gives 173 users in Movie G1.

## Environment

Python 3.8, PyTorch 1.10 (thesis: 1.10, CUDA 11.3), numpy 1.22.0, pandas 1.4.2,
scipy 1.7.3, scikit-learn 1.0.2, setuptools 59.5.0 (newer versions break `import torch`
on 1.10). The thesis lists Python 3.7, but numpy 1.22 requires 3.8.
