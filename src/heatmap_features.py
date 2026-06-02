import numpy as np
import cv2

from configs.config import (
    IMG_SIZE,
    MAX_POINTS,
    HM_PEAK_REL_THRESH,
    HM_PEAK_MIN_DIST,
)

# ======================================================================
# 2.2 Structural features of heatmap point count, coordinates, and spacing
# ======================================================================
def normalize01(x):
    x = x.astype(np.float32)
    if x.max() > x.min():
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return np.zeros_like(x, dtype=np.float32)


def count_heatmap_peaks(hm, rel_thresh=0.30, min_dist=5, max_points=25):
    """
    Detect peak points from the heatmap.
    Return [(x, y, score), ...], sorted from left to right by x coordinate.
    """
    hm = normalize01(hm)
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=1.0, sigmaY=1.0)

    if hm.max() <= 1e-8:
        return []

    thresh = rel_thresh * float(hm.max())
    ys, xs = np.where(hm >= thresh)

    if len(xs) == 0:
        return []

    scores = hm[ys, xs]
    order = np.argsort(-scores)

    selected = []
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        score = float(scores[idx])

        keep = True
        for sx, sy, ss in selected:
            if (x - sx) ** 2 + (y - sy) ** 2 < min_dist ** 2:
                keep = False
                break

        if keep:
            selected.append((x, y, score))

        if len(selected) >= max_points:
            break

    selected = sorted(selected, key=lambda z: z[0])
    return selected


def pad_array(arr, length, value=0.0):
    arr = list(arr)
    if len(arr) >= length:
        return arr[:length]
    return arr + [value] * (length - len(arr))


def basic_stats(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return [0.0] * 9

    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 90)),
    ]


def extract_heatmap_struct_features(hm_224):
    """
    Input:
        hm_224: [224,224], float 0-1

    Output:
        Fixed-length structural features, including:
        - peak_count
        - heatmap global center of mass, variance, and span
        - peak x/y/score sequences
        - adjacent-point dx/dy/dist sequences
        - coordinate, score, and spacing statistics
    """
    hm = normalize01(hm_224)
    H, W = hm.shape

    peaks = count_heatmap_peaks(
        hm,
        rel_thresh=HM_PEAK_REL_THRESH,
        min_dist=HM_PEAK_MIN_DIST,
        max_points=MAX_POINTS
    )

    n = len(peaks)

    xs = np.array([p[0] for p in peaks], dtype=np.float32)
    ys = np.array([p[1] for p in peaks], dtype=np.float32)
    ss = np.array([p[2] for p in peaks], dtype=np.float32)

    xs_n = xs / max(W - 1, 1)
    ys_n = ys / max(H - 1, 1)

    if n >= 2:
        dx = np.diff(xs_n)
        dy = np.diff(ys_n)
        dist = np.sqrt(dx ** 2 + dy ** 2)
    else:
        dx = np.array([], dtype=np.float32)
        dy = np.array([], dtype=np.float32)
        dist = np.array([], dtype=np.float32)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mass = hm.sum() + 1e-8
    cx = float((hm * xx).sum() / mass / max(W - 1, 1))
    cy = float((hm * yy).sum() / mass / max(H - 1, 1))

    vx = float((hm * ((xx / max(W - 1, 1)) - cx) ** 2).sum() / mass)
    vy = float((hm * ((yy / max(H - 1, 1)) - cy) ** 2).sum() / mass)

    if n >= 1:
        span_x = float(xs_n.max() - xs_n.min())
        span_y = float(ys_n.max() - ys_n.min())
    else:
        span_x = 0.0
        span_y = 0.0

    feats = []

    feats.extend([
        float(n) / float(MAX_POINTS),
        float(n),
        float(hm.mean()),
        float(hm.std()),
        float(hm.max()),
        float(hm.sum() / (H * W)),
        cx,
        cy,
        vx,
        vy,
        span_x,
        span_y,
    ])

    feats.extend(pad_array(xs_n.tolist(), MAX_POINTS, 0.0))
    feats.extend(pad_array(ys_n.tolist(), MAX_POINTS, 0.0))
    feats.extend(pad_array(ss.tolist(), MAX_POINTS, 0.0))

    feats.extend(pad_array(dx.tolist(), MAX_POINTS - 1, 0.0))
    feats.extend(pad_array(dy.tolist(), MAX_POINTS - 1, 0.0))
    feats.extend(pad_array(dist.tolist(), MAX_POINTS - 1, 0.0))

    feats.extend(basic_stats(xs_n))
    feats.extend(basic_stats(ys_n))
    feats.extend(basic_stats(ss))
    feats.extend(basic_stats(dx))
    feats.extend(basic_stats(np.abs(dy)))
    feats.extend(basic_stats(dist))

    return np.nan_to_num(np.array(feats, dtype=np.float32))


STRUCT_FEAT_DIM = len(extract_heatmap_struct_features(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)))
