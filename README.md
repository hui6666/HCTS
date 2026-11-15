# HCTS: Open-vocabulary Handwritten Chinese Text Spotting

This repository provides dataset split information and evaluation tools accompanying **Open-vocabulary Handwritten Chinese Text Spotting**, by Zhijie Li, Jiahui Zhang, Dawei Yan, Xing Zhang, Marcin Woźniak, and Wei Dong. The paper has been **accepted for publication in IEEE Transactions on Image Processing (TIP)**.

## HWDB1.0–1.1 datasets and character-class protocol

The controlled open-vocabulary experiments use **3,755 Chinese character classes** from CASIA-HWDB1.0–1.1. The protocol described in the revised manuscript is:

| Split | Character classes | Samples or evaluation images |
| --- | ---: | --- |
| Training | 3,380 seen classes | 90% of the samples within each seen class |
| Validation | The same 3,380 seen classes | Remaining 10% of samples within each seen class; no sample overlap with training |
| Seen test | 3,380 seen classes | 4,000 synthetic full images |
| Unseen test | 375 held-out unseen classes | 1,000 synthetic full images |

The **375 unseen classes (approximately 10% of the 3,755 classes)** are excluded from both training and validation. The unseen full-image test subset contains **21,681 character occurrences** across the 375 held-out classes (**41–82 occurrences per class**). The seen/unseen distinction specifies **character-class membership**; it should not be confused with a guarantee of image- or source-instance-level independence, which must be checked against the underlying data manifests.

Processed HWDB1.0–1.1 dataset resources and seen/unseen character lists are referenced here:

- **V1:** [HWDB1.0–1.1 dataset resources](https://drive.google.com/file/d/12MnjuqaRjPjY5ysVLTgbTAIMGVJ_sYsa/view?usp=sharing)
- **V2:** [HWDB1.0–1.1 dataset resources](https://drive.google.com/file/d/1bWyHe6wBZoQ-hq7Qo8nhsRmeW75-NcFG/view?usp=sharing)

**Scope of this repository:** The source archive contains the CTW **image-ID** split lists and evaluation code; it does not itself include HWDB1.0–1.1 train/validation/test character dictionary files or sample-level manifests. The contents of the linked V1/V2 packages should be checked against the manuscript's stated split before using them for exact reproduction.

## Contents

### `splits/ctw/`

Image-ID lists for the **18,949 CTW images** selected from the original 32,285 images. Each file has one seven-digit CTW image ID per line. The three lists are pairwise disjoint and together cover the selected subset.

| File | Images | Share |
| --- | ---: | ---: |
| `train_ids.txt` | 13,264 | 70% |
| `val_ids.txt` | 1,895 | 10% |
| `test_ids.txt` | 3,790 | 20% |
| **Total** | **18,949** | **100%** |

These files specify **image splits, not character-class dictionaries**. Do not use the CTW image IDs as evidence for the HWDB validation vocabulary.

### `evaluation/`

Self-contained evaluation code using the Python standard library, with support for:

- Detection Precision, Recall and F1, using IoU / IoA / IoD and `ignore` / `difficult` handling.
- Dataset-level 1-NED, alternative 1-NED aggregation methods, and Average Edit Distance (AED).
- Exact-match and character-level recognition accuracy for matched text pairs.

The evaluator takes ground-truth annotations and model predictions as inputs; it does **not** include model weights or an end-to-end inference implementation.

## Usage

Requires **Python 3.8+**.

Run the built-in checks (**68 assertions** in the supplied evaluator):

```bash
cd evaluation
python evaluate.py selftest
```

Evaluate the CTW test split:

```bash
cd evaluation
python evaluate.py eval \
  --gt /path/to/ctw_gt.jsonl \
  --pred /path/to/predictions.jsonl \
  --split-file ../splits/ctw/test_ids.txt \
  --dataset CTW --split test \
  --primary-one-ned dataset_level \
  --report report.json
```

`--gt` and `--pred` accept individual `.jsonl`, `.txt`, `.csv`, `.tsv`, or supported CTW/RCTW-style `.json` files, or directories of per-image `.txt` files. Input formats may be specified with `--gt-format` / `--pred-format`. For available metrics and options, run `python evaluate.py metrics` or `python evaluate.py eval --help`.

| Flag | Default | Description |
| --- | --- | --- |
| `--iou-threshold` | `0.5` | Spatial overlap threshold |
| `--overlap` | `iou` | `iou`, `ioa`, or `iod` |
| `--matching` | `cardinality_iou` | `cardinality_iou`, `greedy`, `score`, or `optimal` |
| `--region-format` | `auto` | Four values represent a rectangle; longer even-length coordinate lists represent polygons |
| `--text-criterion` | `exact` | `any`, `exact`, `ned`, or `substring` |
| `--primary-one-ned` | `dataset_level` | Selection of the primary 1-NED aggregation |

Use `--text-criterion any` for detection-only Precision/Recall/F1. With the default `exact` criterion, the separate end-to-end Precision/Recall/F1 counts depend on correct text as well as spatial correspondence.

## 1-NED evaluation protocol

For a spatially matched, non-ignored ground-truth/prediction text pair \((g_i,p_i)\), instance-level similarity is

\[
\operatorname{1\! -\! NED}_i = 1 - \frac{\operatorname{ED}(g_i,p_i)}{\max(|g_i|,|p_i|)},
\]

where \(\operatorname{ED}\) is Levenshtein edit distance. **The manuscript's dataset-level 1-NED**, and the evaluator's default primary metric, accumulate edit distances and normalization lengths over all spatially matched pairs across the test set:

\[
\operatorname{1\! -\! NED}_{\mathrm{dataset}}
= 1 - \frac{\sum_{i\in M}\operatorname{ED}(g_i,p_i)}
{\sum_{i\in M}\max(|g_i|,|p_i|)},
\]

where \(M\) is the set of eligible spatially matched, non-ignored text pairs. This is a **corpus-level, length-weighted** score, not an average of per-image 1-NED values. If there is no positive normalization length, the evaluator returns no dataset-level score (`null` in JSON).

**Important:** In the released implementation, this `dataset_level` formula does **not directly penalize unmatched ground-truth regions (FN) or unmatched predicted regions (FP)**. Detection Precision/Recall/F1 and penalty-aware 1-NED variants provide complementary information. The phrase *end-to-end* describes evaluation on full-image outputs, but does not imply that this particular 1-NED formula includes an FN/FP penalty.

The evaluator reports the following distinct aggregations:

| Report field (`one_ned_details`) | Computation | Meaning |
| --- | --- | --- |
| `dataset_level` | `1 - sum(ED) / sum(max(len(gt), len(pred)))` on matched pairs | **Manuscript-defined primary metric** |
| `matched_only` | `sum(similarity) / matched` | Unweighted mean on matches |
| `gt_penalized` | `sum(similarity) / (matched + FN)` | Penalizes missed GT regions |
| `pred_penalized` | `sum(similarity) / (matched + FP)` | Penalizes unmatched predictions |
| `full_penalty` | `sum(similarity) / (matched + FN + FP)` | Penalizes both misses and extra predictions |
| `symmetric` | `sum(similarity) / max(matched + FN, matched + FP)` | Symmetric count-based denominator |

Here `matched`, `FN`, and `FP` refer to **spatial detection matching** after ignore-region handling. The separate `one_ned.e2e` field uses the `full_penalty` definition and must **not** be confused with `dataset_level`. The `--text-criterion` argument affects end-to-end Precision/Recall/F1, not the spatial-pair corpus-level 1-NED formula.

For manuscript-style evaluation, use `primary_one_ned.score` or `one_ned_details.dataset_level.score` from the JSON report (multiply by 100 for a percentage). The per-image and macro-averaged outputs are diagnostic and should not be substituted for the corpus-level result. Exact reproduction of publication tables additionally requires the original annotations, predictions, preprocessing, and evaluation settings.

## Citation

```bibtex
@article{hcts2026,
  author  = {Zhijie Li and Jiahui Zhang and Dawei Yan and Xing Zhang and Marcin Woźniak and Wei Dong},
  title   = {Open-vocabulary Handwritten Chinese Text Spotting},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  note    = {Accepted for publication}
}
```

Please update this entry with the official DOI, volume, issue, and page numbers when available.
