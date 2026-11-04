"""
G1a - Geometry as a class cue (TRAIN only, CPU only, no images, no models)
=========================================================================
D:/MARIVISION/H0_workbench/eval_tools/g1a_geometry_class_cue.py

Question (pre-declared in claude/g1a_predeclaration.md):
  Across UNSEEN videos, how much does box geometry alone separate the SeaClips
  classes, and does the curated horizon add anything beyond plain image position?

Feature sets (all normalised by image height H / width W; bottom edge = contact point):
  A  size        : log(h/H), log(w/H), log(w/h)
  B  size+pos    : A + y_bottom/H, y_centre/H, x_centre/W
  C  B+horizon   : B + d_b (y_bottom - curated horizon at box x), d_c, sign(d_b),
                   log(h/H) - log(|d_b| + 0.01)   (the size ratio h/dy),
                   horizon roll (eta_R - eta_L), horizon mean height
  A and B are evaluated on (i) ALL TRAIN boxes and (ii) the g=1 (curated horizon) subset.
  C is evaluated on the g=1 subset only. The C-vs-B comparison is on identical boxes/folds.

Protocol:
  - GroupKFold(5) by video; every box of a video is in exactly one test fold.
  - Learners: HistGradientBoosting (primary) and multinomial LogisticRegression (secondary).
  - Class-balanced training weights. Metric: balanced accuracy over classes present in the
    test fold (computed by hand), plus macro-F1 over present classes.

Pass rules (frozen before any run; see predeclaration):
  R1  geometry is informative     : B balanced acc > chance + 0.10 (all-frame, HGB, pooled over folds)
  R2  position adds to size       : mean(B - A) >= +0.02 and B > A in >= 4/5 folds (all-frame, HGB)
  R3  curated horizon adds        : mean(C - B) >= +0.02 and C > B in >= 4/5 folds (g=1, HGB)
  R1/R2 decide whether the class expert receives box-geometry metadata.
  R3 decides whether any horizon feature is carried forward for classification.

Modes:
  --mode selftest   synthetic checks only (no dataset access)
  --mode run        reads seaclips-train.json + train horizon tracks, writes reports
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "G1a.0"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
TRAIN_TRACKS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/seaclips_horizon_tracks_train_semantic_v2.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools/g1a_reports")

N_FOLDS = 5
EPS_DY = 0.01
MARGIN = 0.02
FOLD_MIN = 4
CHANCE_MARGIN = 0.10

# Expected inventory (from the certified PoL-0 audit)
EXPECT_IMAGES = 20310
EXPECT_ANNS = 65542
EXPECT_G1_IMAGES = 8999
EXPECT_G1_ANNS = 18494


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise AssertionError("[FATAL] " + msg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def norm_key(name):
    s = str(name).replace("\\", "/").strip().lstrip("/")
    if s.startswith("train/"):
        s = s[len("train/"):]
    return s


def load_tracks(path):
    raw = json.load(open(path, "r", encoding="utf-8"))
    flat = {}
    for vid_key, vid in raw.items():
        if not isinstance(vid, dict) or "frames" not in vid:
            continue
        for fname, vec in vid["frames"].items():
            g = float(vec[2])
            require(g in (0.0, 1.0), f"gate value {g} for {fname}")
            flat[norm_key(fname)] = (float(vec[0]), float(vec[1]), g)
    return flat


def load_boxes(train_json, tracks):
    data = json.load(open(train_json, "r", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {im["id"]: im for im in data["images"]}
    rows = []
    missing = 0
    for a in data["annotations"]:
        im = imgs[a["image_id"]]
        key = norm_key(im["file_name"])
        hz = tracks.get(key)
        if hz is None:
            missing += 1
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        rows.append({
            "video": int(im["video_id"]),
            "image_id": a["image_id"],
            "cls": cats[a["category_id"]],
            "W": float(im["width"]), "H": float(im["height"]),
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "eta_L": hz[0], "eta_R": hz[1], "g": hz[2],
            "sensor": im.get("camera_sensor", "unknown"),
        })
    return data, rows, missing


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

FEAT_A = ["log_h", "log_w", "log_aspect"]
FEAT_B = FEAT_A + ["y_bottom", "y_centre", "x_centre"]
FEAT_C = FEAT_B + ["d_bottom", "d_centre", "sign_d", "log_ratio", "roll", "hz_mean"]


def features(r):
    H, W = r["H"], r["W"]
    hn, wn = r["h"] / H, r["w"] / H
    yb = (r["y"] + r["h"]) / H
    yc = (r["y"] + r["h"] / 2.0) / H
    xc = (r["x"] + r["w"] / 2.0) / W
    hz = (1.0 - xc) * r["eta_L"] + xc * r["eta_R"]
    db = yb - hz
    dc = yc - hz
    return {
        "log_h": math.log(hn), "log_w": math.log(wn), "log_aspect": math.log(r["w"] / r["h"]),
        "y_bottom": yb, "y_centre": yc, "x_centre": xc,
        "d_bottom": db, "d_centre": dc, "sign_d": 1.0 if db > 0 else -1.0,
        "log_ratio": math.log(hn) - math.log(abs(db) + EPS_DY),
        "roll": r["eta_R"] - r["eta_L"], "hz_mean": 0.5 * (r["eta_L"] + r["eta_R"]),
    }


def matrix(rows, names):
    F = [features(r) for r in rows]
    return np.array([[f[n] for n in names] for f in F], dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics (hand-written: no sklearn warnings for absent classes)
# ---------------------------------------------------------------------------

def balanced_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    rec = [np.mean(y_pred[y_true == c] == c) for c in classes]
    return float(np.mean(rec)), {str(c): float(v) for c, v in zip(classes, rec)}


def macro_f1(y_true, y_pred):
    out = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def make_folds(groups, k=N_FOLDS):
    """Deterministic GroupKFold-like split: videos sorted by box count, greedy balanced."""
    counts = Counter(groups.tolist())
    order = sorted(counts, key=lambda v: (-counts[v], v))
    load = [0] * k
    assign = {}
    for v in order:
        f = int(np.argmin(load))
        assign[v] = f
        load[f] += counts[v]
    return np.array([assign[v] for v in groups.tolist()])


def class_weights(y):
    cnt = Counter(y.tolist())
    return np.array([1.0 / cnt[c] for c in y.tolist()]) * len(y) / len(cnt)


def make_learner(kind, seed=0):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=seed)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    raise ValueError(kind)


def cross_validate(X, y, folds, kind):
    per_fold = []
    pred_all = np.empty_like(y)
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        model = make_learner(kind)
        w = class_weights(y[tr])
        if kind == "hgb":
            model.fit(X[tr], y[tr], sample_weight=w)
        else:
            model.fit(X[tr], y[tr], logisticregression__sample_weight=w)
        p = model.predict(X[te])
        pred_all[te] = p
        ba, _ = balanced_accuracy(y[te], p)
        per_fold.append({"fold": int(f), "n": int(te.sum()), "classes": int(len(np.unique(y[te]))),
                         "bal_acc": ba, "macro_f1": macro_f1(y[te], p)})
    ba, rec = balanced_accuracy(y, pred_all)
    return {"pooled_bal_acc": ba, "pooled_macro_f1": macro_f1(y, pred_all),
            "per_class_recall": rec, "folds": per_fold}


def paired(res_hi, res_lo):
    d = [a["bal_acc"] - b["bal_acc"] for a, b in zip(res_hi["folds"], res_lo["folds"])]
    return {"fold_deltas": d, "mean_delta": float(np.mean(d)), "n_positive": int(sum(x > 0 for x in d)),
            "pass": bool(np.mean(d) >= MARGIN and sum(x > 0 for x in d) >= FOLD_MIN)}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(rows):
    y_all = np.array([r["cls"] for r in rows])
    v_all = np.array([r["video"] for r in rows])
    folds_all = make_folds(v_all)
    g1 = [r for r in rows if r["g"] == 1.0]
    y_g1 = np.array([r["cls"] for r in g1])
    v_g1 = np.array([r["video"] for r in g1])
    folds_g1 = make_folds(v_g1)

    out = {"inventory": {
        "all_boxes": len(rows), "all_videos": int(len(set(v_all.tolist()))),
        "g1_boxes": len(g1), "g1_videos": int(len(set(v_g1.tolist()))),
        "all_class_counts": dict(Counter(y_all.tolist())), "g1_class_counts": dict(Counter(y_g1.tolist())),
        "all_fold_sizes": dict(Counter(folds_all.tolist())), "g1_fold_sizes": dict(Counter(folds_g1.tolist())),
    }, "results": {}}

    for kind in ("hgb", "logreg"):
        res = {}
        res["all_A"] = cross_validate(matrix(rows, FEAT_A), y_all, folds_all, kind)
        res["all_B"] = cross_validate(matrix(rows, FEAT_B), y_all, folds_all, kind)
        res["g1_A"] = cross_validate(matrix(g1, FEAT_A), y_g1, folds_g1, kind)
        res["g1_B"] = cross_validate(matrix(g1, FEAT_B), y_g1, folds_g1, kind)
        res["g1_C"] = cross_validate(matrix(g1, FEAT_C), y_g1, folds_g1, kind)
        res["delta_all_B_minus_A"] = paired(res["all_B"], res["all_A"])
        res["delta_g1_B_minus_A"] = paired(res["g1_B"], res["g1_A"])
        res["delta_g1_C_minus_B"] = paired(res["g1_C"], res["g1_B"])
        out["results"][kind] = res

    h = out["results"]["hgb"]
    n_cls_all = len(set(y_all.tolist()))
    chance = 1.0 / n_cls_all
    out["verdict"] = {
        "chance_all": chance,
        "R1_geometry_informative": bool(h["all_B"]["pooled_bal_acc"] > chance + CHANCE_MARGIN),
        "R2_position_adds_to_size": h["delta_all_B_minus_A"]["pass"],
        "R3_curated_horizon_adds": h["delta_g1_C_minus_B"]["pass"],
    }
    return out


def write_reports(out, meta):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out["_meta"] = meta
    jp = OUT_DIR / "g1a_results.json"
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2)
    lines = [f"# G1a results ({VERSION})", "", "| Learner | Set | Boxes | Pooled bal. acc | Macro-F1 | Fold bal. acc |",
             "|---|---|---|---|---|---|"]
    for kind, res in out["results"].items():
        for key in ("all_A", "all_B", "g1_A", "g1_B", "g1_C"):
            r = res[key]
            n = sum(f["n"] for f in r["folds"])
            fb = ", ".join(f"{f['bal_acc']:.3f}" for f in r["folds"])
            lines.append(f"| {kind} | {key} | {n} | {r['pooled_bal_acc']:.4f} | {r['pooled_macro_f1']:.4f} | {fb} |")
    lines += ["", "| Learner | Comparison | Mean Δ | Folds > 0 | Pass |", "|---|---|---|---|---|"]
    for kind, res in out["results"].items():
        for key in ("delta_all_B_minus_A", "delta_g1_B_minus_A", "delta_g1_C_minus_B"):
            d = res[key]
            lines.append(f"| {kind} | {key} | {d['mean_delta']:+.4f} | {d['n_positive']}/{len(d['fold_deltas'])} | {d['pass']} |")
    lines += ["", "Per-class recall (HGB, all_B): " + json.dumps(out["results"]["hgb"]["all_B"]["per_class_recall"]),
              "", "Verdict: " + json.dumps(out["verdict"])]
    mp = OUT_DIR / "g1a_results_tables.md"
    open(mp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return jp, mp


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def synth_rows(n_videos=20, per_video=300, seed=0, ratio_signal=True, horizon_varies=True):
    """Synthetic world: class sets physical height; image height = k*phys*dy (pinhole),
    horizon wanders per video when horizon_varies. Class is identifiable from h/dy."""
    rng = np.random.default_rng(seed)
    classes = ["BOAT", "SHIP", "MARINE_MARKER"]
    phys = {"BOAT": 1.0, "SHIP": 4.0, "MARINE_MARKER": 0.4}
    rows = []
    for v in range(n_videos):
        eta = 0.5 + (rng.uniform(-0.15, 0.15) if horizon_varies else 0.0)
        k = rng.uniform(0.15, 0.25)
        for i in range(per_video):
            c = classes[rng.integers(0, 3)]
            dy = rng.uniform(0.02, 0.4)
            hn = (k * phys[c] * dy if ratio_signal else k * rng.uniform(0.3, 4.0) * dy) * rng.lognormal(0, 0.1)
            H, W = 960.0, 1200.0
            yb = eta + dy
            h = hn * H
            w = h * rng.uniform(0.8, 1.5)
            x = rng.uniform(0, W - w)
            rows.append({"video": v, "image_id": v * 10000 + i, "cls": c, "W": W, "H": H,
                         "x": x, "y": yb * H - h, "w": w, "h": h,
                         "eta_L": eta, "eta_R": eta, "g": 1.0, "sensor": "synthetic"})
    return rows


def selftest():
    # 1. Feature geometry on a hand-computed box
    r = {"video": 1, "image_id": 1, "cls": "BOAT", "W": 1000.0, "H": 1000.0, "x": 490.0, "y": 560.0,
         "w": 20.0, "h": 40.0, "eta_L": 0.40, "eta_R": 0.60, "g": 1.0, "sensor": "t"}
    f = features(r)
    require(abs(f["y_bottom"] - 0.60) < 1e-12, "y_bottom")
    require(abs(f["x_centre"] - 0.50) < 1e-12, "x_centre")
    require(abs(f["d_bottom"] - 0.10) < 1e-12, "roll-corrected horizon at x=0.5 must be 0.5")
    require(abs(f["log_ratio"] - (math.log(0.04) - math.log(0.11))) < 1e-12, "log_ratio")
    require(f["sign_d"] == 1.0 and abs(f["roll"] - 0.2) < 1e-12, "sign/roll")

    # 2. Metrics
    yt = np.array(["a", "a", "b", "b"]); yp = np.array(["a", "b", "b", "b"])
    ba, _ = balanced_accuracy(yt, yp)
    require(abs(ba - 0.75) < 1e-12, "balanced accuracy")
    require(abs(macro_f1(yt, yp) - np.mean([2 / 3, 0.8])) < 1e-12, "macro F1")

    # 3. Folds are video-disjoint and deterministic
    g = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 7])
    f1, f2 = make_folds(g), make_folds(g)
    require((f1 == f2).all(), "fold determinism")
    for v in set(g.tolist()):
        require(len(set(f1[g == v].tolist())) == 1, "video split across folds")

    # 4. Positive control: varying horizon + ratio signal -> C must beat B
    rows = synth_rows(horizon_varies=True, ratio_signal=True)
    y = np.array([r["cls"] for r in rows]); v = np.array([r["video"] for r in rows]); fo = make_folds(v)
    rB = cross_validate(matrix(rows, FEAT_B), y, fo, "hgb")
    rC = cross_validate(matrix(rows, FEAT_C), y, fo, "hgb")
    dCB = paired(rC, rB)
    require(dCB["pass"], f"positive control C>B failed: {dCB}")

    # 5. Negative control: fixed horizon at 0.5 -> horizon adds nothing beyond position
    rows = synth_rows(horizon_varies=False, ratio_signal=True, seed=1)
    y = np.array([r["cls"] for r in rows]); v = np.array([r["video"] for r in rows]); fo = make_folds(v)
    rB = cross_validate(matrix(rows, FEAT_B), y, fo, "hgb")
    rC = cross_validate(matrix(rows, FEAT_C), y, fo, "hgb")
    dCB = paired(rC, rB)
    require(not dCB["pass"], f"negative control: fixed horizon must not pass R3: {dCB}")

    # 6. Null control: no geometry signal -> near chance
    rows = synth_rows(horizon_varies=True, ratio_signal=False, seed=2)
    y = np.array([r["cls"] for r in rows]); v = np.array([r["video"] for r in rows]); fo = make_folds(v)
    rB = cross_validate(matrix(rows, FEAT_B), y, fo, "hgb")
    require(rB["pooled_bal_acc"] < 1 / 3 + CHANCE_MARGIN, f"null control above chance+margin: {rB['pooled_bal_acc']}")

    # 7. Logistic learner path runs
    cross_validate(matrix(rows[:600], FEAT_A), y[:600], make_folds(v[:600]), "logreg")
    print(f"[PASS] {VERSION} selftest: features, metrics, video-disjoint folds, positive/negative/null controls")


def run():
    require(TRAIN_JSON.exists(), f"missing {TRAIN_JSON}")
    require(TRAIN_TRACKS.exists(), f"missing {TRAIN_TRACKS}")
    tracks = load_tracks(TRAIN_TRACKS)
    data, rows, missing = load_boxes(TRAIN_JSON, tracks)
    require(len(data["images"]) == EXPECT_IMAGES, f"images {len(data['images'])}")
    require(len(data["annotations"]) == EXPECT_ANNS, f"annotations {len(data['annotations'])}")
    require(missing == 0, f"{missing} annotations without a horizon record")
    g1_imgs = {r["image_id"] for r in rows if r["g"] == 1.0}
    g1_boxes = sum(1 for r in rows if r["g"] == 1.0)
    require(len(g1_imgs) <= EXPECT_G1_IMAGES, f"g1 images {len(g1_imgs)}")
    require(g1_boxes == EXPECT_G1_ANNS, f"g1 boxes {g1_boxes} != {EXPECT_G1_ANNS}")
    print(f"[+] Inventory PASS: {len(rows)} boxes, {g1_boxes} in g=1 frames")
    out = analyse(rows)
    meta = {"version": VERSION, "train_json_sha256": sha256(TRAIN_JSON),
            "tracks_sha256": sha256(TRAIN_TRACKS), "script_sha256": sha256(__file__),
            "n_folds": N_FOLDS, "margin": MARGIN, "fold_min": FOLD_MIN, "chance_margin": CHANCE_MARGIN}
    jp, mp = write_reports(out, meta)
    print(open(mp, encoding="utf-8").read())
    print(f"[+] Wrote {jp}\n[+] Wrote {mp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run"], required=True)
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run()
