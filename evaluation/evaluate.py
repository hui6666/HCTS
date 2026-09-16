from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .detection import (Instance, Counts, detection_metrics, detection_best_f,
                            match_regions, as_points, polygon_area, region_metric,
                            intersection_area, degenerate_count, prf)
    from .recognition import (NormalizeConfig, one_ned, levenshtein,
                              recognition_metrics, accuracy_metrics, e2e_metrics)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from detection import (Instance, Counts, detection_metrics, detection_best_f,
                           match_regions, as_points, polygon_area, region_metric,
                           intersection_area, degenerate_count, prf)
    from recognition import (NormalizeConfig, one_ned, levenshtein,
                             recognition_metrics, accuracy_metrics, e2e_metrics)

__version__ = "1.4.0"

__all__ = [
    "Instance", "EvalConfig", "NormalizeConfig",
    "evaluate", "load_any", "load_config", "dump_report", "dump_summary_csv",
    "format_summary",
    "one_ned", "levenshtein", "region_metric", "match_regions", "selftest",
]


@dataclass
class EvalConfig:
    iou_threshold: float = 0.5
    overlap_kind: str = "iou"
    matching: str = "cardinality_iou"
    ignore_priority: bool = True
    strict_threshold: bool = False
    text_criterion: str = "exact"
    text_ned_threshold: float = 0.5
    region_format: str = "auto"
    require_gt_text: bool = False
    primary_one_ned: str = "matched_only"
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalConfig":
        d = dict(d)
        norm = d.pop("normalize", None)
        known = {f for f in cls.__dataclass_fields__ if f != "normalize"}
        unknown = set(d) - known - {"name", "description", "notes", "line_format"}
        if unknown:
            raise ValueError("unknown config field(s): %s" % sorted(unknown))
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        if norm:
            cfg.normalize = NormalizeConfig(**norm)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def load_config(path: str) -> Tuple[EvalConfig, Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalConfig.from_dict(raw), raw


def _mk_instance(region: Sequence[float], text: str = "", score: float = 1.0,
                 ignore: bool = False, raw: Any = None) -> Instance:
    return Instance(region=[float(v) for v in region], text=text or "",
                    score=float(score), ignore=bool(ignore), raw=raw)


_REGION_KEYS = ("bbox", "polygon", "points", "box", "region", "poly", "quad")


def _flatten_region(v: Any) -> List[float]:
    flat: List[float] = []
    for e in v:
        if isinstance(e, (list, tuple)):
            flat.extend(float(x) for x in e)
        else:
            flat.append(float(e))
    return flat


def _region_from_item(item: Dict[str, Any]) -> Optional[List[float]]:
    for key in _REGION_KEYS:
        if item.get(key) is not None:
            v = item[key]
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return _flatten_region(v)
    return None


def load_jsonl(path: str) -> Dict[str, List[Instance]]:
    data: Dict[str, List[Instance]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            obj = json.loads(line)
            image_id = str(obj.get("image_id") or obj.get("image") or
                           obj.get("id") or lineno)
            items = (obj.get("instances") or obj.get("items") or
                     obj.get("annotations") or obj.get("lines") or
                     obj.get("objects") or [])
            out: List[Instance] = []
            for it in items:
                region = _region_from_item(it)
                if region is None:
                    raise ValueError("%s:%d instance is missing a bbox/polygon field"
                                     % (path, lineno))
                out.append(_mk_instance(
                    region,
                    text=str(it.get("text", it.get("trans",
                                   it.get("transcription", ""))) or ""),
                    score=it.get("score", it.get("confidence", 1.0)),
                    ignore=bool(it.get("ignore", it.get("difficult", False))),
                    raw=it,
                ))
            data.setdefault(image_id, []).extend(out)
    return data


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def parse_line(fields: Sequence[str], line_format: str = "auto"
               ) -> Tuple[List[float], str, bool]:
    fields = [c.strip() for c in fields]
    n = len(fields)

    if line_format == "text_bbox":
        if n < 5 or not all(_is_num(v) for v in fields[-4:]):
            raise ValueError("text_bbox line must be: text xmin ymin xmax ymax")
        text = " ".join(fields[:-4]).strip()
        return [float(v) for v in fields[-4:]], text, False

    if line_format == "rctw":
        coords = [float(v) for v in fields[:8]]
        rest = fields[8:]
        difficult = bool(rest and rest[0] in ("0", "1") and rest[0] == "1")
        if rest and rest[0] in ("0", "1"):
            rest = rest[1:]
        return coords, ",".join(rest), difficult

    if line_format == "coords_text":
        k = 8 if n >= 8 and _is_num(fields[7]) else 4
        return ([float(v) for v in fields[:k]], ",".join(fields[k:]), False)

    if line_format == "coords":
        return [float(v) for v in fields if _is_num(v)], "", False

    if n >= 5 and not _is_num(fields[0]) and all(_is_num(v) for v in fields[-4:]):
        return [float(v) for v in fields[-4:]], " ".join(fields[:-4]).strip(), False

    coords: List[float] = []
    i = 0
    while i < n and _is_num(fields[i]) and len(coords) < 8:
        coords.append(float(fields[i]))
        i += 1
    rest = list(fields[i:])
    difficult = False

    if rest and len(coords) in (4, 8) and rest[0] in ("0", "1"):
        difficult = rest[0] == "1"
        rest = rest[1:]
    elif rest and not difficult and all(_is_num(x) for x in rest) and len(rest) % 2 == 0:
        coords.extend(float(x) for x in rest)
        rest = []

    return coords, ",".join(rest).strip().strip('"'), difficult


def load_txt_dir(directory: str, line_format: str = "auto") -> Dict[str, List[Instance]]:
    data: Dict[str, List[Instance]] = {}
    for fp in sorted(Path(directory).glob("*.txt")):
        items: List[Instance] = []
        with fp.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip().lstrip("﻿")
                if not line:
                    continue
                fields = line.split(",") if "," in line else line.split()
                coords, text, difficult = parse_line(fields, line_format)
                if not coords:
                    continue
                items.append(_mk_instance(coords, text, 1.0, difficult))
        data[fp.stem] = items
    return data


def load_rctw_json(path: str) -> Dict[str, List[Instance]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    data: Dict[str, List[Instance]] = {}
    for image_id, rec in obj.items():
        boxes = rec.get("bboxes") or []
        scores = rec.get("scores") or [1.0] * len(boxes)
        words = rec.get("words") or [""] * len(boxes)
        items = []
        for i, b in enumerate(boxes):
            flat = _flatten_region(b) if isinstance(b, (list, tuple)) else [float(b)]
            items.append(_mk_instance(
                flat,
                str(words[i]) if i < len(words) and words[i] is not None else "",
                scores[i] if i < len(scores) else 1.0,
            ))
        data[str(image_id)] = items
    return data


def load_single_txt(path: str, line_format: str = "auto") -> Dict[str, List[Instance]]:
    data: Dict[str, List[Instance]] = {}
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            fields = line.split(",") if "," in line else line.split()
            coords, text, difficult = parse_line(fields, line_format)
            if coords:
                data.setdefault(Path(path).stem, []).append(
                    _mk_instance(coords, text, 1.0, difficult))
    return data


def load_any(path: str, fmt: str = "auto", line_format: str = "auto"
             ) -> Dict[str, List[Instance]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("input not found: %s" % p)
    if fmt == "auto":
        if p.is_dir():
            fmt = "dir"
        elif p.suffix.lower() == ".jsonl":
            fmt = "jsonl"
        elif p.suffix.lower() == ".json":
            fmt = "rctw_json"
        elif p.suffix.lower() in (".txt", ".csv", ".tsv"):
            fmt = "txt"
        else:
            raise ValueError(
                "cannot infer format from the suffix, pass --gt-format: %s" % p)
    if fmt == "dir":
        return load_txt_dir(path, line_format)
    if fmt == "jsonl":
        return load_jsonl(path)
    if fmt == "rctw_json":
        return load_rctw_json(path)
    if fmt == "txt":
        return load_single_txt(path, line_format)
    raise ValueError(
        "unknown input format: %r (expected jsonl / txt / dir / rctw_json)" % (fmt,))


def load_split_ids(path: str) -> List[str]:
    ids: List[str] = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for lineno, line in enumerate(stream, 1):
            image_id = line.strip()
            if not image_id:
                continue
            if image_id in seen:
                raise ValueError("duplicate image_id at line %d of the split file: %s"
                                 % (lineno, image_id))
            seen.add(image_id)
            ids.append(image_id)
    if not ids:
        raise ValueError("split file is empty: %s" % path)
    return ids


def restrict_to_split(gt_map: Dict[str, List[Instance]],
                      pred_map: Dict[str, List[Instance]],
                      split_ids: Sequence[str]
                      ) -> Tuple[Dict[str, List[Instance]], Dict[str, List[Instance]]]:
    missing_gt = [image_id for image_id in split_ids if image_id not in gt_map]
    if missing_gt:
        raise ValueError("split has %d image(s) without GT, e.g. %s" %
                         (len(missing_gt), missing_gt[:5]))
    return ({image_id: list(gt_map[image_id]) for image_id in split_ids},
            {image_id: list(pred_map.get(image_id, [])) for image_id in split_ids})


def path_sha256(path: str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(source.read_bytes())
        return digest.hexdigest()
    for file_path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(source)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit_inputs(gt_map: Dict[str, List[Instance]],
                 pred_map: Dict[str, List[Instance]],
                 cfg: Optional[EvalConfig] = None,
                 max_examples: int = 3) -> List[str]:
    cfg = cfg or EvalConfig()
    warn: List[str] = []
    G = [x for v in gt_map.values() for x in v]
    P = [x for v in pred_map.values() for x in v]

    for name, items in (("GT", G), ("prediction", P)):
        bad = degenerate_count(items, cfg.region_format)
        if bad and items:
            pct = bad / len(items) * 100
            warn.append(
                "%s has %d/%d (%.1f%%) zero-area regions. Typical causes: 4-value "
                "bbox parsed as a polygon (region_format=%s), or duplicated/"
                "collinear vertices." % (
                    name, bad, len(items), pct, cfg.region_format))
            if cfg.region_format == "polygon" and any(
                    len(x.region) == 4 for x in items):
                warn.append(
                    "  -> 4-value coordinates with region_format=polygon. "
                    "If these are x1,y1,x2,y2 rectangles, use "
                    "region_format=auto or bbox.")

    empty_p = sum(1 for x in P if not str(x.text).strip())
    if P and empty_p == len(P):
        warn.append(
            "All prediction instances have empty text, so the end-to-end and "
            "1-NED numbers are meaningless. To evaluate detection only, pass "
            "--text-criterion any explicitly.")
    elif empty_p:
        warn.append("%d/%d prediction instances have empty text." % (empty_p, len(P)))

    only_gt = set(gt_map) - set(pred_map)
    only_pred = set(pred_map) - set(gt_map)
    if only_gt and only_pred and not (set(gt_map) & set(pred_map)):
        warn.append(
            "GT and prediction image_id sets are disjoint (GT %d, prediction %d). "
            "Examples: GT=%s vs prediction=%s" % (
                len(gt_map), len(pred_map),
                sorted(only_gt)[:max_examples], sorted(only_pred)[:max_examples]))
    elif only_gt or only_pred:
        warn.append(
            "image_id sets are not aligned: %d only in GT, %d only in prediction "
            "(misses and false positives on those images count in full)."
            % (len(only_gt), len(only_pred)))

    if G and len(P) > 3 * len(G):
        warn.append(
            "Prediction count (%d) is %.1f x GT (%d); this looks like missing NMS "
            "or duplicated output, and will sharply lower precision."
            % (len(P), len(G), len(P) / len(G)))

    empty_g = sum(1 for x in G if not str(x.text).strip() and not x.ignore)
    if G and empty_g:
        warn.append(
            "%d GT instances have empty text and are not marked ignore; they will "
            "always count as misses." % empty_g)

    return warn


def evaluate(gt_map: Dict[str, List[Instance]],
             pred_map: Dict[str, List[Instance]],
             cfg: Optional[EvalConfig] = None) -> Dict[str, Any]:
    cfg = cfg or EvalConfig()
    ids = sorted(set(gt_map) | set(pred_map))
    all_gt = [x for values in gt_map.values() for x in values]
    all_pred = [x for values in pred_map.values() for x in values]
    if cfg.require_gt_text:
        empty = sum(1 for x in all_gt if not x.ignore and not str(x.text).strip())
        if empty:
            raise ValueError(
                "%d non-ignore GT instances have empty text; the E2E-1-NED "
                "protocol cannot be computed" % empty)

    det_cnt = Counts()
    e2e_cnt = Counts()
    per_image: List[Dict[str, Any]] = []
    sim_sum = 0.0
    edit_sum = 0.0
    all_gt_text_len = 0.0
    matched_gt_text_len = 0.0
    unmatched_pred_text_len = 0.0
    exact_count = 0
    n_text_matches = 0

    def add_counts(dst: Counts, src: Counts) -> None:
        for key in ("n_gt", "n_pred", "tp", "fp", "fn",
                    "ignored_gt", "dropped_pred"):
            setattr(dst, key, getattr(dst, key) + getattr(src, key))

    for image_id in ids:
        g = list(gt_map.get(image_id, []))
        p = list(pred_map.get(image_id, []))
        _det_img, cnt, m = detection_metrics(
            g, p, cfg.iou_threshold, cfg.overlap_kind,
            cfg.matching, cfg.strict_threshold, cfg.region_format,
            ignore_priority=cfg.ignore_priority)
        _e2e_img, e_cnt, _ = e2e_metrics(
            g, p, cfg.iou_threshold, cfg.overlap_kind,
            cfg.matching, cfg.strict_threshold, cfg.text_criterion,
            cfg.text_ned_threshold, cfg.normalize, cfg.region_format,
            ignore_priority=cfg.ignore_priority)
        add_counts(det_cnt, cnt)
        add_counts(e2e_cnt, e_cnt)

        image_sim_sum = 0.0
        image_text_matches = 0
        dropped_pred = {pi for gi, pi, _v in m.pairs if g[gi].ignore}
        for gi, row in enumerate(m.matrix):
            if g[gi].ignore:
                for pi in m.unmatched_pred:
                    if row[pi] >= 0.5:
                        dropped_pred.add(pi)

        for inst in g:
            if not inst.ignore:
                all_gt_text_len += len(cfg.normalize.apply(inst.text))
        for gi, pi, _overlap in m.pairs:
            if g[gi].ignore:
                continue
            gt_text = cfg.normalize.apply(g[gi].text)
            pred_text = cfg.normalize.apply(p[pi].text)
            similarity = one_ned(gt_text, pred_text)
            sim_sum += similarity
            image_sim_sum += similarity
            edit_sum += levenshtein(gt_text, pred_text)
            matched_gt_text_len += len(gt_text)
            exact_count += int(gt_text == pred_text)
            n_text_matches += 1
            image_text_matches += 1
        for pi in m.unmatched_pred:
            if pi not in dropped_pred:
                unmatched_pred_text_len += len(cfg.normalize.apply(p[pi].text))

        image_denominator = cnt.tp + cnt.fn + cnt.fp
        image_e2e_ned = (image_sim_sum / image_denominator
                         if image_denominator else 1.0)
        per_image.append({
            "image_id": image_id,
            "gt": cnt.n_gt,
            "pred": cnt.n_pred,
            "matched": cnt.tp,
            "tp": cnt.tp,
            "fn": cnt.fn,
            "fp": cnt.fp,
            "dropped_pred": cnt.dropped_pred,
            "e2e_one_ned": image_e2e_ned,
            "e2e_one_ned_numerator": image_sim_sum,
            "e2e_one_ned_denominator": image_denominator,
            "one_ned_matched_only": (image_sim_sum / image_text_matches
                                     if image_text_matches else 0.0),
        })

    det = {**prf(det_cnt.tp, det_cnt.fp, det_cnt.fn)}
    det.update({key + "_percent": value * 100.0 for key, value in list(det.items())})
    e2e = {**prf(e2e_cnt.tp, e2e_cnt.fp, e2e_cnt.fn)}
    e2e.update({key + "_percent": value * 100.0 for key, value in list(e2e.items())})

    canonical_denominator = n_text_matches + det_cnt.fn + det_cnt.fp
    canonical = sim_sum / canonical_denominator if canonical_denominator else 1.0
    effective_predictions = n_text_matches + det_cnt.fp
    rec = {
        "e2e": canonical,
        "e2e_percent": canonical * 100.0,
        "numerator": sim_sum,
        "denominator": canonical_denominator,
        "n_matched": n_text_matches,
        "n_fn": det_cnt.fn,
        "n_fp": det_cnt.fp,
        "matched_only": sim_sum / n_text_matches if n_text_matches else 0.0,
        "gt_penalized": sim_sum / det_cnt.n_gt if det_cnt.n_gt else 1.0,
        "pred_penalized": (sim_sum / effective_predictions
                           if effective_predictions else 1.0),
        "symmetric": (sim_sum / max(det_cnt.n_gt, effective_predictions)
                      if max(det_cnt.n_gt, effective_predictions) else 1.0),
        "aed": (edit_sum + (all_gt_text_len - matched_gt_text_len) +
                unmatched_pred_text_len) / len(ids) if ids else 0.0,
        "aed_avg_gt_len": (all_gt_text_len / det_cnt.n_gt
                           if det_cnt.n_gt else 0.0),
    }
    for key in ("matched_only", "gt_penalized", "pred_penalized", "symmetric"):
        rec[key + "_percent"] = rec[key] * 100.0
    acc = {
        "n_matched": n_text_matches,
        "exact_match": exact_count / n_text_matches if n_text_matches else 0.0,
        "exact_match_percent": (exact_count / n_text_matches * 100.0
                                if n_text_matches else 0.0),
        "char_level": sim_sum / n_text_matches if n_text_matches else 0.0,
        "char_level_percent": (sim_sum / n_text_matches * 100.0
                               if n_text_matches else 0.0),
    }

    macro: Dict[str, float] = {}
    if per_image:
        for key in ("one_ned_matched_only", "e2e_one_ned"):
            macro[key] = sum(r[key] for r in per_image) / len(per_image)
        macro["f1"] = sum(prf(r["tp"], r["fp"], r["fn"])["f1"]
                          for r in per_image) / len(per_image)

    def aggregate_details(numerator, matched, fn, fp):
        denominators = {"matched_only": matched, "gt_penalized": matched + fn,
                        "pred_penalized": matched + fp,
                        "full_penalty": matched + fn + fp,
                        "symmetric": max(matched + fn, matched + fp)}
        return {name: {"numerator": numerator, "denominator": den,
                       "score": numerator / den if den else None,
                       "percent": 100 * numerator / den if den else None}
                for name, den in denominators.items()}

    details = aggregate_details(sim_sum, n_text_matches, det_cnt.fn, det_cnt.fp)

    valid_primary = {"matched_only", "gt_penalized", "pred_penalized",
                     "full_penalty", "symmetric"}
    if cfg.primary_one_ned not in valid_primary:
        raise ValueError("unknown primary_one_ned %r; choose one of %s" %
                         (cfg.primary_one_ned, sorted(valid_primary)))

    m_count = n_text_matches
    upper_bounds = {
        "matched_only": 1.0 if m_count else None,
        "gt_penalized": (m_count / (m_count + det_cnt.fn)
                         if (m_count + det_cnt.fn) else None),
        "pred_penalized": (m_count / (m_count + det_cnt.fp)
                           if (m_count + det_cnt.fp) else None),
        "full_penalty": (m_count / (m_count + det_cnt.fn + det_cnt.fp)
                         if (m_count + det_cnt.fn + det_cnt.fp) else None),
        "symmetric": (m_count / max(m_count + det_cnt.fn,
                                    m_count + det_cnt.fp)
                      if max(m_count + det_cnt.fn,
                             m_count + det_cnt.fp) else None),
    }
    primary_detail = dict(details[cfg.primary_one_ned])
    primary_detail["name"] = cfg.primary_one_ned
    primary_detail["upper_bound"] = upper_bounds[cfg.primary_one_ned]
    primary_detail["upper_bound_percent"] = (
        100.0 * upper_bounds[cfg.primary_one_ned]
        if upper_bounds[cfg.primary_one_ned] is not None else None)

    for row in per_image:
        row["one_ned_details"] = aggregate_details(
            row["e2e_one_ned_numerator"], row["matched"], row["fn"], row["fp"])
    macro_details = {}
    for name in details:
        values = [row["one_ned_details"][name]["score"] for row in per_image
                  if row["one_ned_details"][name]["score"] is not None]
        macro_details[name] = {"score": sum(values) / len(values) if values else None,
                               "percent": 100 * sum(values) / len(values) if values else None,
                               "included_images": len(values),
                               "excluded_images": len(per_image) - len(values)}

    return {
        "primary_one_ned": primary_detail,
        "one_ned_upper_bounds": {
            k: {"score": v, "percent": (100.0 * v if v is not None else None)}
            for k, v in upper_bounds.items()
        },
        "one_ned_details": details,
        "one_ned_macro_details": macro_details,
        "config": cfg.to_dict(),
        "input_warnings": audit_inputs(gt_map, pred_map, cfg),
        "counts": {**det_cnt.as_dict(), "images": len(ids)},
        "detection": det,
        "end_to_end": e2e,
        "one_ned": rec,
        "accuracy": acc,
        "macro_average": macro,
        "per_image": per_image,
    }


def dump_report(report: Dict[str, Any], path: Optional[str] = None,
                indent: int = 2) -> str:
    text = json.dumps(report, ensure_ascii=False, indent=indent)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    return text


def dump_summary_csv(report: Dict[str, Any], path: str,
                     dataset: str, split: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    det = report["detection"]
    names = list(report["one_ned_details"])
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "split", "n_images", "precision", "recall", "f1",
                         "primary_one_ned_name", "primary_one_ned_percent"] +
                        ["one_ned_" + name + "_micro_percent" for name in names] +
                        ["one_ned_" + name + "_macro_percent" for name in names])
        values = [report["one_ned_details"][name]["percent"] for name in names] + [
            report["one_ned_macro_details"][name]["percent"] for name in names]
        primary = report["primary_one_ned"]
        writer.writerow([dataset, split, report["counts"]["images"],
                         det["precision_percent"], det["recall_percent"], det["f1_percent"],
                         primary["name"],
                         "" if primary["percent"] is None else primary["percent"]] +
                        ["" if v is None else v for v in values])


def format_summary(report: Dict[str, Any]) -> str:
    c = report.get("counts", {})
    d = report.get("detection", {})
    e = report.get("end_to_end", {})
    n = report.get("one_ned", {})
    a = report.get("accuracy", {})
    m = report.get("macro_average", {})
    cfg = report.get("config", {})
    L = ["=" * 68]
    L.append("images %s | GT %s | pred %s | matched %s" % (
        c.get("images"), c.get("n_gt"), c.get("n_pred"), n.get("n_matched")))
    L.append("TP %s | FP %s | FN %s | dropped(ignore) %s" % (
        c.get("tp"), c.get("fp"), c.get("fn"), c.get("dropped_pred")))
    L.append("protocol: IoU>=%.2f (%s) | matching=%s%s | text=%s" % (
        cfg.get("iou_threshold", 0), cfg.get("overlap_kind", ""),
        cfg.get("matching", ""),
        " strict>" if cfg.get("strict_threshold") else "",
        cfg.get("text_criterion", "")))
    L.append("-" * 68)
    L.append("detection   P=%6.2f  R=%6.2f  F1=%6.2f" % (
        d.get("precision_percent", 0), d.get("recall_percent", 0),
        d.get("f1_percent", 0)))
    L.append("end-to-end  P=%6.2f  R=%6.2f  F1=%6.2f" % (
        e.get("precision_percent", 0), e.get("recall_percent", 0),
        e.get("f1_percent", 0)))
    L.append("-" * 68)
    primary = report.get("primary_one_ned", {})
    pscore = primary.get("percent")
    pub = primary.get("upper_bound_percent")
    L.append("primary 1-NED (%s): %s%s" % (
        primary.get("name", ""),
        "N/A" if pscore is None else "%.2f%%" % pscore,
        "" if pub is None else "  [count-based upper bound %.2f%%]" % pub))
    L.append("1-NED: independent aggregation definitions (percent)")
    for name, value in report["one_ned_details"].items():
        ma = report["one_ned_macro_details"][name]
        score = "N/A" if value["percent"] is None else "%.2f" % value["percent"]
        mac = "N/A" if ma["percent"] is None else "%.2f" % ma["percent"]
        L.append("  %s: micro=%s [%g/%d]; macro=%s (%d included, %d excluded images)" %
                 (name, score, value["numerator"], value["denominator"], mac,
                  ma["included_images"], ma["excluded_images"]))
    L.append("Exact-match accuracy on matched regions: %.2f%%" % a.get("exact_match_percent", 0))
    L.append("=" * 68)
    return "\n".join(L)


def selftest(verbose: bool = True) -> int:
    fails: List[str] = []
    n_checks = [0]

    def check(name, got, want, tol=1e-6):
        n_checks[0] += 1
        if abs(float(got) - float(want)) > tol:
            fails.append("%s: got %r, want %r" % (name, got, want))

    check("rect IoU (partial)", region_metric([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)
    check("rect IoU (self)", region_metric([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    check("rect IoU (disjoint)", region_metric([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
    check("IoA", region_metric([0, 0, 10, 10], [5, 5, 15, 15], "ioa"), 0.25)
    check("IoD", region_metric([0, 0, 10, 10], [5, 5, 15, 15], "iod"), 0.25)
    quad = [0, 0, 10, 0, 10, 10, 0, 10]
    check("quad area", polygon_area(as_points(quad)), 100.0)
    check("quad self IoU", region_metric(quad, quad), 1.0)
    diamond = [5, -5, 15, 5, 5, 15, -5, 5]
    check("diamond area", polygon_area(as_points(diamond)), 200.0)
    check("diamond-square IoU", region_metric([0, 0, 10, 10], diamond), 100 / 200)
    lshape = [0, 0, 10, 0, 10, 5, 5, 5, 5, 10, 0, 10]
    check("concave polygon area", polygon_area(as_points(lshape)), 75.0)
    check("concave polygon intersection", intersection_area(as_points(lshape),
                                                            as_points([2, 2, 8, 8])), 27.0)
    check("concave polygon IoU", region_metric(lshape, [2, 2, 8, 8]), 27 / 84)

    deg4 = [_mk_instance([0, 0, 10, 10], "a")]
    check("degenerate IoU must be 0", region_metric(deg4[0].region, deg4[0].region,
                                                    "iou", "polygon"), 0.0)
    check("degenerate count", degenerate_count(deg4, "polygon"), 1)
    check("non-degenerate count", degenerate_count(deg4, "auto"), 0)
    w = audit_inputs({"i": deg4}, {"i": deg4}, EvalConfig(region_format="polygon"))
    n_checks[0] += 1
    if not w:
        fails.append("degenerate regions were not reported by audit_inputs")

    check("lev identical", levenshtein("abc", "abc"), 0)
    check("lev substitution", levenshtein("abc", "abd"), 1)
    check("lev CJK", levenshtein("手写汉字识别", "手写汉子识别"), 1)
    check("1-NED identical", one_ned("手写汉字识别", "手写汉字识别"), 1.0)
    check("1-NED one error", one_ned("手写汉字识别", "手写汉子识别"), 1 - 1 / 6, 1e-9)
    check("1-NED all wrong", one_ned("abcdef", "uvwxyz"), 0.0)
    check("1-NED both empty", one_ned("", ""), 1.0)
    check("1-NED empty vs non-empty", one_ned("", "abc"), 0.0)
    check("equal after normalization", one_ned("A B c", "abc",
                                               NormalizeConfig(strip_spaces=True, casefold=True)), 1.0)

    gt = {
        "img_001": [_mk_instance([10, 10, 110, 50], "手写汉字识别"),
                    _mk_instance([10, 70, 90, 110], "开放词汇")],
        "img_002": [_mk_instance([20, 20, 120, 60], "文本检测")],
    }
    pred = {
        "img_001": [_mk_instance([12, 11, 109, 49], "手写汉子识别")],
        "img_002": [_mk_instance([19, 19, 121, 61], "文本检测"),
                    _mk_instance([150, 150, 220, 190], "误检文本")],
    }
    rep = evaluate(gt, pred, EvalConfig(iou_threshold=0.5, region_format="bbox"))
    check("regression detection P", rep["detection"]["precision"], 2 / 3, 1e-9)
    check("regression detection R", rep["detection"]["recall"], 2 / 3, 1e-9)
    check("regression detection F1", rep["detection"]["f1"], 2 / 3, 1e-9)
    check("regression matched_only", rep["one_ned"]["matched_only"], 0.9166666667, 1e-9)
    check("regression gt_penalized", rep["one_ned"]["gt_penalized"], 0.6111111111, 1e-9)
    check("regression symmetric", rep["one_ned"]["symmetric"], 0.6111111111, 1e-9)
    check("regression TP", rep["counts"]["tp"], 2)
    check("regression FP", rep["counts"]["fp"], 1)
    check("regression FN", rep["counts"]["fn"], 1)

    cross_gt = {"A": [_mk_instance([0, 0, 10, 10], "甲")]}
    cross_pred = {"B": [_mk_instance([0, 0, 10, 10], "甲")]}
    cross = evaluate(cross_gt, cross_pred, EvalConfig(region_format="bbox"))
    check("cross-image match forbidden TP", cross["counts"]["tp"], 0)
    check("cross-image match forbidden FP", cross["counts"]["fp"], 1)
    check("cross-image match forbidden FN", cross["counts"]["fn"], 1)
    check("cross-image match forbidden F1", cross["detection"]["f1"], 0.0)
    check("cross-image match forbidden E2E-1-NED", cross["one_ned"]["e2e"], 0.0)

    card_gt = [_mk_instance([0, 0, 10, 10], "甲"),
               _mk_instance([2, 0, 12, 10], "乙")]
    card_pred = [_mk_instance([0, 0, 10, 10], "乙"),
                 _mk_instance([-2, 0, 8, 10], "甲")]
    greedy_match = match_regions(card_gt, card_pred, 0.5, strategy="greedy")
    card_match = match_regions(card_gt, card_pred, 0.5, strategy="cardinality_iou")
    check("greedy counterexample pair count", len(greedy_match.pairs), 1)
    check("max-cardinality-first pair count", len(card_match.pairs), 2)
    check("default matcher maximizes count", len(match_regions(card_gt, card_pred).pairs), 2)
    default_report = evaluate({"i": card_gt}, {"i": card_pred})
    check("default evaluation match count", default_report["counts"]["tp"], 2)
    check("perfect matched recognition reaches bound", default_report["one_ned"]["e2e"], 1.0)

    sym_gt = {"i": [_mk_instance([0, 0, 10, 10], "甲"),
                    _mk_instance([20, 0, 30, 10], "乙")]}
    sym_pred = {"i": [_mk_instance([0, 0, 10, 10], "甲"),
                      _mk_instance([40, 0, 50, 10], "丙")]}
    sym = evaluate(sym_gt, sym_pred, EvalConfig(
        region_format="bbox", matching="cardinality_iou", require_gt_text=True))
    check("E2E-1-NED numerator", sym["one_ned"]["numerator"], 1.0)
    check("E2E-1-NED denominator", sym["one_ned"]["denominator"], 3)
    check("E2E-1-NED symmetric penalty", sym["one_ned"]["e2e"], 1 / 3)

    simple_g = [_mk_instance([0, 0, 10, 10], "a")]
    simple_p = [_mk_instance([0, 0, 10, 10], "a", 0.9),
                _mk_instance([0, 0, 10, 10], "a", 0.5)]
    for strat in ("greedy", "score", "optimal"):
        m = match_regions(simple_g, simple_p, 0.5, "iou", strat)
        check("strategy %s pair count" % strat, len(m.pairs), 1)

    g2 = [_mk_instance([0, 0, 10, 10], "a"), _mk_instance([20, 0, 30, 10], "b")]
    p2 = [_mk_instance([0, 0, 10, 10], "a", 0.9),
          _mk_instance([100, 0, 110, 10], "x", 0.1)]
    bf = detection_best_f(g2, p2, 0.5)
    check("best-F precision", bf["precision"], 1.0, 1e-9)
    _e, c_e, _m = e2e_metrics(g2, p2, text_criterion="exact")
    check("e2e TP", c_e.tp, 1)
    _e2, c_bad, _m2 = e2e_metrics(g2, [_mk_instance([0, 0, 10, 10], "z", 0.9)],
                                  text_criterion="exact")
    check("e2e wrong text -> FP", c_bad.fp, 1)
    check("e2e wrong text -> FN", c_bad.fn, 2)

    g3 = [_mk_instance([0, 0, 10, 10], "a", 1.0, True),
          _mk_instance([20, 0, 30, 10], "b")]
    p3 = [_mk_instance([0, 0, 10, 10], "zz")]
    _d3, c3, _m3 = detection_metrics(g3, p3, 0.5)
    check("ignore: n_gt", c3.n_gt, 1)
    check("ignore: hits on ignore are not FP", c3.fp, 0)
    check("ignore: dropped count", c3.dropped_pred, 1)

    c, t, d = parse_line("1,2,3,4,5,6,7,8".split(","))
    check("line coords-only count", len(c), 8)
    check("line coords-only first value", c[0], 1.0)
    if t != "":
        fails.append("line coords-only should have no text: got %r" % t)
    c, t, d = parse_line("1,2,3,4,5,6,7,8,手写汉字识别".split(","))
    check("line coords+text count", len(c), 8)
    if t != "手写汉字识别":
        fails.append("line coords+text: got %r" % t)
    c, t, d = parse_line("1,2,3,4,5,6,7,8,1,文本".split(","))
    check("line rctw difficult", d, True)
    if t != "文本":
        fails.append("line rctw text: got %r" % t)
    c, t, d = parse_line("1,2,3,4,5,6,7,8,0,文本".split(","))
    check("line rctw difficult=0", d, False)
    c, t, d = parse_line("琉奶顺 230 265 440 351".split(), "text_bbox")
    check("line text_bbox coord count", len(c), 4)
    check("line text_bbox xmin", c[0], 230)
    if t != "琉奶顺":
        fails.append("line text_bbox text: got %r" % t)

    n_checks[0] += 1
    if fails:
        print("selftest failed %d / %d checks:" % (len(fails), n_checks[0]))
        for f in fails:
            print("  x %s" % f)
        return 1
    if verbose:
        print("selftest passed  (%d assertions)" % n_checks[0])
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Text detection/recognition metrics toolkit "
                    "(P/R/F1, 1-NED, AED, accuracy)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="compute metrics")
    e.add_argument("--gt", required=True,
                   help="GT: jsonl / txt / rctw json / directory")
    e.add_argument("--pred", required=True, help="predictions: same as GT")
    e.add_argument("--config",
                   help="configs/*.json preset (CLI flags override its fields)")
    e.add_argument("--gt-format", default="auto",
                   choices=["auto", "jsonl", "txt", "dir", "rctw_json"])
    e.add_argument("--pred-format", default="auto",
                   choices=["auto", "jsonl", "txt", "dir", "rctw_json"])
    e.add_argument("--line-format", default=None,
                   choices=["auto", "coords", "coords_text", "rctw", "text_bbox"],
                   help="field layout for line-based txt (default auto)")
    e.add_argument("--iou-threshold", type=float, help="default 0.5")
    e.add_argument("--overlap", choices=["iou", "ioa", "iod"])
    e.add_argument("--matching",
                   choices=["greedy", "score", "optimal", "cardinality_iou"],
                   help="default cardinality_iou: maximize match count, then total overlap")
    e.add_argument("--legacy-ignore-matching", action="store_true",
                   help="allow ignored GT to compete for matches (v1.2 behavior)")
    e.add_argument("--strict", action="store_true",
                   help="require overlap strictly greater than the threshold "
                        "(default >=)")
    e.add_argument("--region-format", choices=["auto", "bbox", "polygon"],
                   help="auto: 4 values = rectangle, otherwise polygon")
    e.add_argument("--text-criterion", choices=["any", "exact", "ned", "substring"],
                   help="end-to-end criterion: any=boxes only | exact=full string "
                        "| ned=1-NED>=threshold | substring=containment")
    e.add_argument("--text-ned-threshold", type=float)
    e.add_argument("--primary-one-ned",
                   choices=["matched_only", "gt_penalized", "pred_penalized",
                            "full_penalty", "symmetric"],
                   help="publication-facing 1-NED field; default matched_only. "
                        "All definitions are always reported.")
    e.add_argument("--strip-spaces", action="store_true")
    e.add_argument("--casefold", action="store_true")
    e.add_argument("--fullwidth-to-halfwidth", action="store_true")
    e.add_argument("--strip-quotes", action="store_true")
    e.add_argument("--report", help="output path for the JSON report")
    e.add_argument("--summary-csv", help="output explicitly named micro/macro 1-NED columns")
    e.add_argument("--dataset", help="dataset name in the CSV/report, e.g. CTW")
    e.add_argument("--split", help="split name in the CSV/report, e.g. test")
    e.add_argument("--split-file",
                   help="one image_id per line; evaluate only this subset")
    e.add_argument("--quiet", action="store_true",
                   help="write output/files only, no printed summary")
    e.set_defaults(func=_cmd_eval)

    sub.add_parser("selftest",
                   help="run built-in numeric assertions").set_defaults(
        func=lambda a: selftest())
    sub.add_parser("metrics",
                   help="list supported metrics and strategies").set_defaults(
        func=_cmd_metrics)
    return p


def _cmd_eval(args: argparse.Namespace) -> int:
    cfg = EvalConfig()
    preset_name = None
    raw: Dict[str, Any] = {}
    if args.config:
        cfg, raw = load_config(args.config)
        preset_name = raw.get("name") or Path(args.config).stem
    overrides = {
        "iou_threshold": args.iou_threshold,
        "overlap_kind": args.overlap,
        "matching": args.matching,
        "text_criterion": args.text_criterion,
        "text_ned_threshold": args.text_ned_threshold,
        "region_format": args.region_format,
        "primary_one_ned": args.primary_one_ned,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.legacy_ignore_matching:
        cfg.ignore_priority = False
    if args.strict:
        cfg.strict_threshold = True
    for flag, attr in (("strip_spaces", "strip_spaces"), ("casefold", "casefold"),
                       ("fullwidth_to_halfwidth", "fullwidth_to_halfwidth"),
                       ("strip_quotes", "strip_quotes")):
        if getattr(args, flag):
            setattr(cfg.normalize, attr, True)

    line_format = args.line_format or raw.get("line_format", "auto")
    gt = load_any(args.gt, args.gt_format, line_format)
    pred = load_any(args.pred, args.pred_format, line_format)
    split_ids = None
    if args.split_file:
        split_ids = load_split_ids(args.split_file)
        gt, pred = restrict_to_split(gt, pred, split_ids)
    report = evaluate(gt, pred, cfg)
    if preset_name:
        report["config"]["preset"] = preset_name
    dataset = args.dataset or preset_name or ""
    split = args.split or (Path(args.split_file).stem if args.split_file else "")
    report["evaluation"] = {
        "dataset": dataset,
        "split": split,
        "split_file": str(Path(args.split_file).resolve()) if args.split_file else None,
        "split_file_sha256": path_sha256(args.split_file) if args.split_file else None,
        "gt": str(Path(args.gt).resolve()),
        "gt_sha256": path_sha256(args.gt),
        "prediction": str(Path(args.pred).resolve()),
        "prediction_sha256": path_sha256(args.pred),
        "evaluator_version": __version__,
        "aggregation": "micro over instances after per-image matching",
        "primary_one_ned": cfg.primary_one_ned,
        "one_ned_outputs": "primary_one_ned plus all definitions in one_ned_details and one_ned_macro_details",
    }
    for w in report.get("input_warnings", []):
        print("[input check] %s" % w, file=sys.stderr)
    if not args.quiet:
        print(format_summary(report))
    if args.report:
        dump_report(report, args.report)
        if not args.quiet:
            print("report written to %s" % args.report)
    elif args.quiet:
        print(dump_report(report))
    if args.summary_csv:
        dump_summary_csv(report, args.summary_csv, dataset, split)
        if not args.quiet:
            print("metric summary CSV written to %s" % args.summary_csv)
    return 0


def _cmd_metrics(_args: argparse.Namespace) -> int:
    print("""Supported metrics and strategies
======================================================================
Region overlap  region_metric(a, b, kind)          detection.py
                kind: iou | ioa | iod
                4 values = [x1,y1,x2,y2] rectangle; 8 values = quad;
                2N values = N-gon. Any simple polygon is supported
                (ear-clipped internally, convex-convex takes a fast path).

Matching        match_regions(gt, pred, iou_threshold, kind, strategy)
                greedy   candidate pairs by descending overlap, greedy 1-to-1
                score    predictions by descending confidence, each takes the
                         best available GT (ICDAR / IC15 style)
                optimal  Hungarian algorithm, global optimum

Detection       detection_metrics(...)   -> P / R / F1 (honours ignore/difficult)
                detection_best_f(...)    -> best-F over a confidence sweep

End-to-end      e2e_metrics(..., text_criterion)
                any | exact | ned | substring

Recognition     one_ned(gt, pred)          recognition.py
                instance-level 1-NED = 1 - ED / max(|gt|, |pred|)
                recognition_metrics(...)   dataset-level aggregates:
                matched_only / gt_penalized / pred_penalized / full_penalty / symmetric / aed
                accuracy_metrics(...)      exact-match rate / char-level accuracy

One-stop        evaluate(gt_map, pred_map, EvalConfig)      evaluate.py
                load_any(path) supports jsonl / txt / dir / rctw_json
======================================================================""")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
