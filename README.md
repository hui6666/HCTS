# HCTS: Open-vocabulary Handwritten Chinese Text Spotting

This repository provides supplementary resources for the manuscript **Open-vocabulary Handwritten Chinese Text Spotting** by Zhijie Li, Jiahui Zhang, Dawei Yan, Xing Zhang, Marcin Woźniak, and Wei Dong.

## HWDB1.0–1.1 datasets and class splits

The following resources are provided to support reproducibility: the processed HWDB1.0–1.1 subsets used in our experiments and the exact seen/unseen character lists for the open-vocabulary evaluation. The corresponding dataset resources are available at the following links:

- **V1:** [Download the HWDB1.0–1.1 dataset resources](https://drive.google.com/file/d/12MnjuqaRjPjY5ysVLTgbTAIMGVJ_sYsa/view?usp=sharing)
- **V2:** [Download the HWDB1.0–1.1 dataset resources](https://drive.google.com/file/d/1bWyHe6wBZoQ-hq7Qo8nhsRmeW75-NcFG/view?usp=sharing)

These resources accompany the HWDB1.0–1.1 experiments described in the manuscript. The CTW image split lists and evaluation scripts are provided separately in this repository, as described below.

## Contents

- `splits/ctw/`
  The image IDs of the 18,949 CTW images used in our experiments, selected from the
  original 32,285 images. One image ID per line (7-digit CTW ID); the three files are
  disjoint and their union is the full subset.

  | File | Images | Share |
  | --- | ---: | ---: |
  | `train_ids.txt` | 13,264 | 70% |
  | `val_ids.txt` | 1,895 | 10% |
  | `test_ids.txt` | 3,790 | 20% |
  | **Total** | **18,949** | 100% |

- `evaluation/`
  A self-contained metrics toolkit (Python standard library only, no third-party
  dependencies) covering:
  - Detection Precision, Recall and F1 score, with IoU / IoA / IoD matching and
    `ignore` / `difficult` handling
  - End-to-End 1-NED and Average Edit Distance (AED)
  - Exact-match and character-level recognition accuracy on matched pairs

## Usage

Requires Python 3.8 or newer.

Verify the evaluator first — it ships with 62 built-in numeric assertions:

```bash
cd evaluation
python evaluate.py selftest
```

Run an evaluation on the test split:

```bash
cd evaluation
python evaluate.py eval \
  --gt   /path/to/ctw_gt.jsonl \
  --pred /path/to/predictions.jsonl \
  --split-file ../splits/ctw/test_ids.txt \
  --dataset CTW --split test \
  --report report.json
```

`--gt` and `--pred` accept either a single file (`.jsonl`, `.txt`, `.csv`, `.tsv`,
CTW/RCTW-style `.json`) or a directory of per-image `.txt` files; the format is
inferred from the suffix and can be forced with `--gt-format` / `--pred-format`.

Protocol switches (see `python evaluate.py metrics` for the full list):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--iou-threshold` | `0.5` | overlap threshold for a match |
| `--overlap` | `iou` | `iou` / `ioa` / `iod` |
| `--matching` | `greedy` | `greedy` / `score` / `optimal` / `cardinality_iou` |
| `--region-format` | `auto` | 4 values = rectangle, otherwise polygon |
| `--text-criterion` | `exact` | `any` / `exact` / `ned` / `substring` |

For **detection-only** evaluation, pass `--text-criterion any`; otherwise a predicted
box whose text does not match counts as both a false positive and a false negative.

The evaluator reports five 1-NED aggregations at once (E2E-1-NED, `matched_only`,
`gt_penalized`, `pred_penalized`, `symmetric`) because the choice of denominator
matters. The primary number used in the paper is **E2E-1-NED**, computed as
`sum(similarity) / (matched + FN + FP)`.


## Citation

```bibtex
@unpublished{hcts2026,
  author = {Zhijie Li and Jiahui Zhang and Dawei Yan and Xing Zhang and Marcin Woźniak and Wei Dong},
  title  = {Open-vocabulary Handwritten Chinese Text Spotting},
  year   = {2026},
  note   = {Unpublished manuscript}
}
```
