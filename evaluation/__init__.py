from .detection import (
    Instance, MatchResult, Counts,
    as_points, polygon_area, signed_area, is_convex, intersection_area,
    region_metric, degenerate_count, match_regions, detection_metrics,
    detection_best_f, prf,
)
from .recognition import (
    NormalizeConfig, normalize_text, levenshtein, one_ned, char_accuracy,
    e2e_metrics, recognition_metrics, accuracy_metrics,
)
from .evaluate import (
    EvalConfig, evaluate, audit_inputs, load_any, load_jsonl, load_txt_dir,
    load_rctw_json, load_config, dump_report, dump_summary_csv, format_summary,
    parse_line, load_split_ids, restrict_to_split, selftest,
)

__version__ = "1.5.0"

__all__ = [
    "Instance", "MatchResult", "Counts",
    "as_points", "polygon_area", "signed_area", "is_convex", "intersection_area",
    "region_metric", "degenerate_count", "match_regions", "detection_metrics",
    "detection_best_f", "prf",
    "NormalizeConfig", "normalize_text", "levenshtein", "one_ned", "char_accuracy",
    "e2e_metrics", "recognition_metrics", "accuracy_metrics",
    "EvalConfig", "evaluate", "audit_inputs", "load_any", "load_jsonl",
    "load_txt_dir", "load_rctw_json", "load_config", "dump_report",
    "dump_summary_csv", "format_summary", "parse_line", "load_split_ids",
    "restrict_to_split", "selftest",
]
