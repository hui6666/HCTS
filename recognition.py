from __future__ import annotations

import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .detection import (Instance, MatchResult, Counts,
                            match_regions, prf, with_percent)
except ImportError:
    from detection import (Instance, MatchResult, Counts,
                           match_regions, prf, with_percent)

__all__ = [
    "NormalizeConfig", "normalize_text", "levenshtein", "one_ned", "char_accuracy",
    "e2e_metrics", "recognition_metrics", "accuracy_metrics",
]


_CJK_PUNCT_MAP = str.maketrans({
    "，": ",", "。": ".", "、": ",", "：": ":", "；": ";",
    "？": "?", "！": "!", "（": "(", "）": ")", "【": "[", "】": "]",
    "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "－": "-", "　": " ",
})
_QUOTE_MAP = str.maketrans({c: None for c in "\"'“”‘’《》「」『』"})
_ASCII_PUNCT_MAP = str.maketrans({c: None for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"})


@dataclass
class NormalizeConfig:
    strip_spaces: bool = False
    casefold: bool = False
    fullwidth_to_halfwidth: bool = False
    strip_quotes: bool = False
    remove_punct: bool = False
    unify_cjk_punct: bool = False

    def apply(self, s) -> str:
        if s is None:
            return ""
        s = str(s)
        if self.fullwidth_to_halfwidth:
            s = unicodedata.normalize("NFKC", s)
        if self.unify_cjk_punct:
            s = s.translate(_CJK_PUNCT_MAP)
        if self.strip_quotes:
            s = s.translate(_QUOTE_MAP)
        if self.remove_punct:
            s = s.translate(_ASCII_PUNCT_MAP)
        if self.strip_spaces:
            s = "".join(s.split())
        if self.casefold:
            s = s.casefold()
        return s

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


def normalize_text(s, cfg: Optional[NormalizeConfig] = None) -> str:
    return cfg.apply(s) if cfg else str(s)


def levenshtein(a: str, b: str,
                ins: float = 1.0, dele: float = 1.0, sub: float = 1.0) -> float:
    if a == b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
        ins, dele = dele, ins
    prev = [j * ins for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [i * dele]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + ins,
                prev[j] + dele,
                prev[j - 1] + (0.0 if ca == cb else sub),
            ))
        prev = cur
    return float(prev[-1])


def one_ned(gt: str, pred: str, norm: Optional[NormalizeConfig] = None) -> float:
    if norm is not None:
        gt, pred = norm.apply(gt), norm.apply(pred)
    gt, pred = str(gt), str(pred)
    den = max(len(gt), len(pred))
    if den == 0:
        return 1.0
    return 1.0 - levenshtein(gt, pred) / den


def char_accuracy(gt: str, pred: str, norm: Optional[NormalizeConfig] = None) -> float:
    return one_ned(gt, pred, norm)


def _text_ok(gt_text: str, pred_text: str, criterion: str,
             ned_threshold: float, norm: Optional[NormalizeConfig]) -> bool:
    g = norm.apply(gt_text) if norm else str(gt_text)
    p = norm.apply(pred_text) if norm else str(pred_text)
    if criterion == "any":
        return True
    if criterion == "exact":
        return g == p
    if criterion == "ned":
        return one_ned(g, p) >= ned_threshold
    if criterion == "substring":
        return (g in p) or (p in g)
    raise ValueError(
        "unknown text criterion: %r (expected any/exact/ned/substring)" % (criterion,))


def e2e_metrics(gt: Sequence[Instance], pred: Sequence[Instance],
                iou_threshold: float = 0.5, kind: str = "iou",
                strategy: str = "cardinality_iou", strict: bool = False,
                text_criterion: str = "exact", text_ned_threshold: float = 0.5,
                norm: Optional[NormalizeConfig] = None,
                region_fmt: str = "auto", ignore_priority: bool = True
                ) -> Tuple[Dict[str, float], object, MatchResult]:
    match = match_regions(gt, pred, iou_threshold, kind, strategy, strict, region_fmt, ignore_priority)
    c = Counts()
    c.n_gt = sum(1 for g in gt if not g.ignore)
    c.ignored_gt = len(gt) - c.n_gt
    c.n_pred = len(pred)
    for gi, pi, _v in match.pairs:
        if gt[gi].ignore:
            c.dropped_pred += 1
            continue
        if _text_ok(gt[gi].text, pred[pi].text, text_criterion, text_ned_threshold, norm):
            c.tp += 1
        else:
            c.fp += 1
            c.fn += 1
    ignore_hit = set()
    for gi, row in enumerate(match.matrix):
        if gt[gi].ignore:
            for j in match.unmatched_pred:
                if row[j] >= 0.5:
                    ignore_hit.add(j)
    c.dropped_pred += len(ignore_hit)
    c.fp += len([j for j in match.unmatched_pred if j not in ignore_hit])
    c.fn += len([i for i in match.unmatched_gt if not gt[i].ignore])
    return with_percent(prf(c.tp, c.fp, c.fn)), c, match


def recognition_metrics(gt: Sequence[Instance], pred: Sequence[Instance],
                        match: Optional[MatchResult] = None,
                        iou_threshold: float = 0.5, kind: str = "iou",
                        strategy: str = "cardinality_iou", strict: bool = False,
                        norm: Optional[NormalizeConfig] = None,
                        n_images: int = 1,
                        region_fmt: str = "auto") -> Dict[str, float]:
    if match is None:
        match = match_regions(gt, pred, iou_threshold, kind, strategy, strict, region_fmt)

    n_gt = sum(1 for g in gt if not g.ignore)
    n_pred = len(pred)

    matched_sims: List[float] = []
    edit_total = 0.0
    matched_norm_len = 0.0
    matched_gt_len = 0.0
    all_gt_len = 0.0
    for g in gt:
        if not g.ignore:
            all_gt_len += len(normalize_text(g.text, norm))
    for gi, pi, _v in match.pairs:
        if gt[gi].ignore:
            continue
        g_t = normalize_text(gt[gi].text, norm)
        p_t = normalize_text(pred[pi].text, norm)
        matched_sims.append(one_ned(g_t, p_t))
        edit_total += levenshtein(g_t, p_t)
        matched_norm_len += max(len(g_t), len(p_t))
        matched_gt_len += len(g_t)

    sim_sum = sum(matched_sims)
    n_match = len(matched_sims)

    aed_num = edit_total + (all_gt_len - matched_gt_len)
    for j in match.unmatched_pred:
        aed_num += len(normalize_text(pred[j].text, norm))

    out = {
        "dataset_level": (1.0 - edit_total / matched_norm_len
                          if matched_norm_len > 0 else 0.0),
        "matched_only": sim_sum / n_match if n_match else 0.0,
        "gt_penalized": sim_sum / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0),
        "pred_penalized": sim_sum / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0),
        "symmetric": sim_sum / max(n_gt, n_pred) if max(n_gt, n_pred) else 1.0,
        "aed": aed_num / n_images if n_images > 0 else 0.0,
        "n_matched": n_match,
        "aed_avg_gt_len": all_gt_len / n_gt if n_gt else 0.0,
    }
    for k in ("dataset_level", "matched_only", "gt_penalized",
              "pred_penalized", "symmetric"):
        out[k + "_percent"] = out[k] * 100.0
    return out


def accuracy_metrics(gt: Sequence[Instance], pred: Sequence[Instance],
                     match: Optional[MatchResult] = None,
                     iou_threshold: float = 0.5, kind: str = "iou",
                     strategy: str = "cardinality_iou", strict: bool = False,
                     norm: Optional[NormalizeConfig] = None,
                     region_fmt: str = "auto") -> Dict[str, float]:
    if match is None:
        match = match_regions(gt, pred, iou_threshold, kind, strategy, strict, region_fmt)
    exact, chars, n = 0, 0.0, 0
    for gi, pi, _v in match.pairs:
        if gt[gi].ignore:
            continue
        g_t = normalize_text(gt[gi].text, norm)
        p_t = normalize_text(pred[pi].text, norm)
        n += 1
        if g_t == p_t:
            exact += 1
        chars += one_ned(g_t, p_t)
    return {
        "n_matched": n,
        "exact_match": exact / n if n else 0.0,
        "exact_match_percent": (exact / n * 100.0) if n else 0.0,
        "char_level": chars / n if n else 0.0,
        "char_level_percent": (chars / n * 100.0) if n else 0.0,
    }
