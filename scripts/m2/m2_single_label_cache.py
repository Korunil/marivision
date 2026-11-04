r"""m2_single_label_cache.py - CPU only. Writes a "single label per query" copy of an M1.1 tile cache. VAL only.
MARIVISION-HR-P2, Claude lead, 2026-10-28.

Deformable-DETR's post-processor (R0) returns the top-100 (query, class) PAIRS of each image/tile, so one query can
appear several times with the SAME box and different classes (e.g. SAILING_VESSEL 0.9 and BOAT 0.3). Every extra copy
is a wrong-class false positive on a real object. The oracle headroom check showed wrong-class FPs are where most of
the recoverable AP is.

PRE-DECLARED RULE "SL" (2026-10-28, parameter-free, applied identically to every model and variant):
  within each tile's raw R0 output, entries with bit-identical boxes come from the same query; keep only the
  highest-scoring entry of each such group (ties: the earlier entry, i.e. R0's own rank order). Nothing else changes:
  plain = Stage B after SL; tiled = V0 merge after SL, both scored with m2_evaluate.py unchanged.
  Primary: T_tiled vs F_plain (all SL), offset 0, M1 success rule. Replication: offset 5.

The output cache keeps the input's schema and meta (plus a "single_label" block with counts and the source cache SHA),
so every existing tool reads it unchanged. The input cache is never modified.

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m2_single_label_cache.py --mode selftest
  python -W error m2_single_label_cache.py --mode convert --cache <r0_tile_cache_....json.gz>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

SCRIPT_VERSION = "SL.0"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def single_label(dets):
    """dets: [[score, label, x1, y1, x2, y2], ...] in R0's rank order -> one entry per distinct box (max score)."""
    best = {}
    order = []
    for i, d in enumerate(dets):
        key = tuple(d[2:6])
        if key not in best:
            best[key] = i
            order.append(key)
        elif d[0] > dets[best[key]][0]:
            best[key] = i
    keep = sorted(best[k] for k in order)
    return [dets[i] for i in keep]


def selftest() -> None:
    dets = [[0.9, 5, 10, 10, 50, 50], [0.3, 1, 10, 10, 50, 50], [0.5, 2, 100, 100, 110, 106],
            [0.6, 3, 100, 100, 110, 106], [0.2, 2, 300, 300, 310, 306]]
    out = single_label(dets)
    assert out == [[0.9, 5, 10, 10, 50, 50], [0.6, 3, 100, 100, 110, 106], [0.2, 2, 300, 300, 310, 306]], out
    assert single_label([]) == []
    near = [[0.9, 5, 10, 10, 50, 50], [0.8, 1, 10.001, 10, 50, 50]]
    assert len(single_label(near)) == 2  # only bit-identical boxes are the same query
    print("  [PASS] one entry per identical box, highest score kept, distinct boxes untouched, rank order preserved")
    print(f"[SL SELFTEST] 1/1 passed (script {SCRIPT_VERSION})")


def convert(a) -> None:
    src = Path(a.cache).resolve()
    if not src.name.endswith(".json.gz") or "_SL" in src.name:
        raise SystemExit("[REFUSED] expected an original r0_tile_cache_*.json.gz")
    dst = src.with_name(src.name.replace("_stride", "_SL_stride"))
    if dst.exists():
        raise SystemExit(f"[REFUSED] {dst} exists.")
    with gzip.open(src, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    n_in = n_out = 0
    for fr in cache["frames"].values():
        for t in fr["tiles"]:
            n_in += len(t[5])
            t[5] = single_label(t[5])
            n_out += len(t[5])
    cache["meta"]["single_label"] = {"script_version": SCRIPT_VERSION, "source_cache": str(src),
                                     "source_cache_sha256": sha256_file(src), "entries_in": n_in, "entries_out": n_out}
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"[SL] {src.name}: {n_in} raw entries -> {n_out} ({1 - n_out / max(1, n_in):.1%} were extra class labels of a box)")
    print(f"[SL] wrote {dst}\n     sha256 {sha256_file(dst)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "convert"), required=True)
    ap.add_argument("--cache")
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else convert(a)


if __name__ == "__main__":
    main()
