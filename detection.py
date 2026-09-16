from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Region = Sequence[float]

__all__ = [
    "Point", "Region", "Instance", "MatchResult", "Counts",
    "as_points", "polygon_area", "signed_area", "is_convex",
    "intersection_area", "region_metric",
    "match_regions", "detection_metrics", "detection_best_f",
]


def as_points(region: Region, fmt: str = "auto") -> List[Point]:
    vals = [float(v) for v in region]
    if fmt == "bbox" or (fmt == "auto" and len(vals) == 4):
        if len(vals) != 4:
            raise ValueError("bbox requires 4 values, got %d" % len(vals))
        x1, y1, x2, y2 = vals
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if len(vals) % 2 != 0:
        raise ValueError("polygon vertex coordinates must be an even count, got %d" % len(vals))
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def signed_area(poly: Sequence[Point]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def polygon_area(poly: Sequence[Point]) -> float:
    return abs(signed_area(poly))


def ensure_ccw(poly: Sequence[Point]) -> List[Point]:
    p = list(poly)
    return p if signed_area(p) >= 0 else p[::-1]


def is_convex(poly: Sequence[Point]) -> bool:
    n = len(poly)
    if n < 4:
        return True
    sign = 0
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-12:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _seg_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _clip_convex(subject: Sequence[Point], clip_ccw: Sequence[Point]) -> List[Point]:
    output = list(subject)
    if not output:
        return []
    n = len(clip_ccw)
    for i in range(n):
        ax, ay = clip_ccw[i]
        bx, by = clip_ccw[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        inp, output = output, []
        if not inp:
            break
        prev = inp[-1]
        prev_in = (ex * (prev[1] - ay) - ey * (prev[0] - ax)) >= -1e-12
        for cur in inp:
            cur_in = (ex * (cur[1] - ay) - ey * (cur[0] - ax)) >= -1e-12
            if cur_in:
                if not prev_in:
                    output.append(_seg_intersect(prev, cur, (ax, ay), (bx, by)))
                output.append(cur)
            elif prev_in:
                output.append(_seg_intersect(prev, cur, (ax, ay), (bx, by)))
            prev, prev_in = cur, cur_in
    return output


def _point_in_tri(p: Point, a: Point, b: Point, c: Point) -> bool:
    def cr(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3 = cr(a, b, p), cr(b, c, p), cr(c, a, p)
    neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (neg and pos)


def _ear_clip(poly: Sequence[Point]) -> List[Tuple[Point, Point, Point]]:
    pts = ensure_ccw(poly)
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(pts[0], pts[1], pts[2])]
    idx = list(range(n))
    tris: List[Tuple[Point, Point, Point]] = []
    guard = 0
    while len(idx) > 3 and guard < 3 * n * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) <= 1e-12:
                continue
            if any(_point_in_tri(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((a, b, c))
            del idx[k]
            clipped = True
            break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append((pts[idx[0]], pts[idx[1]], pts[idx[2]]))
    return tris


def _convex_pieces(poly: Sequence[Point]) -> List[Sequence[Point]]:
    if is_convex(poly):
        return [ensure_ccw(poly)]
    return list(_ear_clip(poly))


def intersection_area(a: Sequence[Point], b: Sequence[Point]) -> float:
    pieces_a = _convex_pieces(a)
    pieces_b = _convex_pieces(b)
    if not pieces_a or not pieces_b:
        return 0.0
    total = 0.0
    for pa in pieces_a:
        for pb in pieces_b:
            clipped = _clip_convex(pa, pb)
            if len(clipped) >= 3:
                total += polygon_area(clipped)
    return total


def region_metric(a: Region, b: Region, kind: str = "iou", fmt: str = "auto") -> float:
    pa, pb = as_points(a, fmt), as_points(b, fmt)
    aa, ab = polygon_area(pa), polygon_area(pb)
    if aa <= 0 or ab <= 0:
        return 0.0
    inter = intersection_area(pa, pb)
    if kind == "iou":
        union = aa + ab - inter
        return 0.0 if union <= 0 else inter / union
    if kind == "ioa":
        return 0.0 if aa <= 0 else inter / aa
    if kind == "iod":
        return 0.0 if ab <= 0 else inter / ab
    raise ValueError("unknown overlap kind: %r (expected iou / ioa / iod)" % (kind,))


@dataclass
class Instance:
    region: Region
    text: str = ""
    score: float = 1.0
    ignore: bool = False
    raw: Any = None

    def poly(self, fmt: str = "auto") -> List[Point]:
        return as_points(self.region, fmt)


@dataclass
class MatchResult:
    pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    unmatched_gt: List[int] = field(default_factory=list)
    unmatched_pred: List[int] = field(default_factory=list)
    matrix: List[List[float]] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return len(self.pairs)

    def pair_map(self) -> Dict[int, int]:
        return {gi: pi for gi, pi, _ in self.pairs}


@dataclass
class Counts:
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignored_gt: int = 0
    dropped_pred: int = 0

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def _hungarian_min(cost: List[List[float]]) -> List[Tuple[int, int]]:
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return []
    size = max(n, m)
    big = (max(max(r) for r in cost) if (n and m) else 1.0) * (size + 1) + 1.0
    a = [[cost[i][j] if (i < n and j < m) else big for j in range(size)]
         for i in range(size)]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out: List[Tuple[int, int]] = []
    for j in range(1, size + 1):
        i = p[j] - 1
        if 0 <= i < n and 0 <= j - 1 < m:
            out.append((i, j - 1))
    return out


def match_regions(gt: Sequence[Instance], pred: Sequence[Instance],
                  iou_threshold: float = 0.5,
                  kind: str = "iou",
                  strategy: str = "cardinality_iou",
                  strict: bool = False,
                  region_fmt: str = "auto",
                  ignore_priority: bool = True) -> MatchResult:
    n, m = len(gt), len(pred)
    if n == 0 or m == 0:
        return MatchResult([], list(range(n)), list(range(m)), [])

    def ok(v: float) -> bool:
        return v > iou_threshold if strict else v >= iou_threshold

    matrix = [[region_metric(g.region, p.region, kind, region_fmt) for p in pred]
              for g in gt]
    if ignore_priority and any(g.ignore for g in gt):
        valid = [i for i, g in enumerate(gt) if not g.ignore]
        sub = match_regions([gt[i] for i in valid], pred, iou_threshold,
                            kind, strategy, strict, region_fmt, False)
        pairs = [(valid[i], j, v) for i, j, v in sub.pairs]
        used_g = {i for i, _, _ in pairs}
        used_p = {j for _, j, _ in pairs}
        return MatchResult(pairs, [i for i in range(n) if i not in used_g],
                           [j for j in range(m) if j not in used_p], matrix)
    res = MatchResult(matrix=matrix)

    if strategy in ("optimal", "cardinality_iou"):
        size = n + m
        cardinality_bonus = min(n, m) + 1.0
        invalid_cost = cardinality_bonus * (size + 1.0)
        cost = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(n):
            for j in range(m):
                cost[i][j] = (-(cardinality_bonus + matrix[i][j])
                              if ok(matrix[i][j]) else invalid_cost)
        assignment = _hungarian_min(cost)
        pairs = [(i, j) for i, j in assignment
                 if i < n and j < m and ok(matrix[i][j])]
    elif strategy == "score":
        used_g: set = set()
        pairs = []
        for j in sorted(range(m), key=lambda j: (-pred[j].score, j)):
            best_i, best_v = -1, -1.0
            for i in range(n):
                if i in used_g:
                    continue
                if ok(matrix[i][j]) and matrix[i][j] > best_v:
                    best_i, best_v = i, matrix[i][j]
            if best_i >= 0:
                used_g.add(best_i)
                pairs.append((best_i, j))
    elif strategy == "greedy":
        cand = [(matrix[i][j], i, j)
                for i in range(n) for j in range(m) if ok(matrix[i][j])]
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        used_g, used_p = set(), set()
        pairs = []
        for _v, i, j in cand:
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            pairs.append((i, j))
    else:
        raise ValueError(
            "unknown matching strategy: %r "
            "(expected greedy / score / optimal / cardinality_iou)"
            % (strategy,))

    pairs = sorted((i, j, matrix[i][j]) for i, j in pairs)
    used_g = {i for i, _, _ in pairs}
    used_p = {j for _, j, _ in pairs}
    res.pairs = pairs
    res.unmatched_gt = [i for i in range(n) if i not in used_g]
    res.unmatched_pred = [j for j in range(m) if j not in used_p]
    return res


def _tally(gt: Sequence[Instance], pred: Sequence[Instance],
           match: MatchResult, drop_iou: float = 0.5) -> Counts:
    c = Counts()
    c.n_gt = sum(1 for g in gt if not g.ignore)
    c.ignored_gt = len(gt) - c.n_gt
    c.n_pred = len(pred)
    for gi, _pi, _v in match.pairs:
        if gt[gi].ignore:
            c.dropped_pred += 1
        else:
            c.tp += 1
    ignore_hit = set()
    for gi, row in enumerate(match.matrix):
        if gt[gi].ignore:
            for j in match.unmatched_pred:
                if row[j] >= drop_iou:
                    ignore_hit.add(j)
    c.dropped_pred += len(ignore_hit)
    c.fp = len([j for j in match.unmatched_pred if j not in ignore_hit])
    c.fn = sum(1 for gi in match.unmatched_gt if not gt[gi].ignore)
    return c


def prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def with_percent(d: Dict[str, float]) -> Dict[str, float]:
    return {**d, **{k + "_percent": v * 100.0 for k, v in d.items()}}


def degenerate_count(instances: Sequence[Instance], fmt: str = "auto") -> int:
    n = 0
    for x in instances:
        try:
            if polygon_area(as_points(x.region, fmt)) <= 0:
                n += 1
        except (ValueError, TypeError):
            n += 1
    return n


def detection_metrics(gt: Sequence[Instance], pred: Sequence[Instance],
                      iou_threshold: float = 0.5, kind: str = "iou",
                      strategy: str = "cardinality_iou", strict: bool = False,
                      region_fmt: str = "auto",
                      with_pct: bool = True, ignore_priority: bool = True
                      ) -> Tuple[Dict[str, float], Counts, MatchResult]:
    match = match_regions(gt, pred, iou_threshold, kind, strategy, strict, region_fmt, ignore_priority)
    cnt = _tally(gt, pred, match)
    out = prf(cnt.tp, cnt.fp, cnt.fn)
    return (with_percent(out) if with_pct else out), cnt, match


def detection_best_f(gt: Sequence[Instance], pred: Sequence[Instance],
                     iou_threshold: float = 0.5, kind: str = "iou",
                     strict: bool = False, region_fmt: str = "auto",
                     min_score: float = 0.0) -> Dict[str, Any]:
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "score_threshold": min_score,
                "precision_percent": 0.0, "recall_percent": 0.0, "f1_percent": 0.0}
    order = sorted(range(len(pred)), key=lambda j: (-pred[j].score, j))
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "score_threshold": min_score}
    for k in range(1, len(order) + 1):
        if pred[order[k - 1]].score < min_score:
            break
        kept = [pred[j] for j in order[:k]]
        m = match_regions(gt, kept, iou_threshold, kind, "score", strict, region_fmt)
        c = _tally(gt, kept, m)
        cur = prf(c.tp, c.fp, c.fn)
        if cur["f1"] > best["f1"]:
            best = {**cur, "score_threshold": pred[order[k - 1]].score}
    return with_percent(best)
