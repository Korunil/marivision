r"""
G5 TEST inventory: how many TEST frames exist, and how long a TEST export will take.  Read-only, CPU, seconds.

Location: anywhere (suggested D:/MARIVISION/H0_workbench/eval_tools/g5_test_inventory.py).
What it reads:  * the zip's file LIST (names and sizes from the central directory; no image is decompressed)
                * the NAMES, sizes and SHA-256 of *.json files in D:/MARIVISION (bytes are hashed, never parsed)
What it never does:  open or parse any annotation file, decode any image, touch a model.
So no TEST label, box or class count is seen; TEST stays locked until the G5 evaluation.

Output: D:/MARIVISION/G5/g5_test_inventory.md and .json (counts only).

Modes:  --mode selftest  |  --mode run
"""
from PIL import Image  # noqa: F401  -- PIL first (Director's Windows env); not used otherwise
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "G5I.0"
ROOT = Path("D:/MARIVISION")
ZIP = ROOT / "SeaClips.zip"
OUT = ROOT / "G5"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# measured rates (G4b exports, RTX 3060, fp32): 17-tile ConvNeXt-B export 826 s / 432 frames
SEC_TILED_CNB = 826.0 / 432
SEC_PLAIN_R50 = 0.15          # one 640x640 R50 inference pass per frame (estimate, not measured)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def split_of(parts):
    for p in parts:
        q = p.lower()
        if q in ("train", "val", "valid", "validation", "test"):
            return q
    return "other:" + (parts[0] if parts else "")


def video_of(parts):
    for p in parts:
        if p.lower().startswith("video_"):
            return p
    return "?"


def inventory(zip_path, json_dir):
    per_split = defaultdict(Counter)
    sizes = Counter()
    other = Counter()
    with zipfile.ZipFile(zip_path, "r") as z:
        infos = z.infolist()
    for zi in infos:
        if zi.is_dir():
            continue
        parts = zi.filename.replace("\\", "/").strip("/").split("/")
        if Path(parts[-1]).suffix.lower() not in IMG_EXT:
            other[Path(parts[-1]).suffix.lower() or "(none)"] += 1
            continue
        s = split_of(parts[:-1])
        per_split[s][video_of(parts[:-1])] += 1
        sizes[s] += zi.compress_size
    jsons = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(Path(json_dir).glob("*.json"))]
    out = {"zip": str(zip_path), "zip_bytes": Path(zip_path).stat().st_size, "splits": {}, "non_image_files": dict(other),
           "json_files_in_root (names only, never parsed)": jsons}
    for s, c in sorted(per_split.items()):
        out["splits"][s] = {"frames": sum(c.values()), "videos": len(c), "frames_per_video": dict(sorted(c.items())),
                            "compressed_gb": sizes[s] / 2 ** 30}
    t = out["splits"].get("test")
    if t:
        n = t["frames"]
        out["test_export_projection_h"] = {"tiled_convnext_b_17_tiles_per_seed": n * SEC_TILED_CNB / 3600,
                                           "r0_plain_once": n * SEC_PLAIN_R50 / 3600,
                                           "three_seeds_plus_r0": (3 * n * SEC_TILED_CNB + n * SEC_PLAIN_R50) / 3600}
    return out


def report(o):
    L = [f"# G5 TEST inventory ({VERSION}) -- zip file list only, no labels read", "",
         f"Zip: {o['zip']} ({o['zip_bytes'] / 2 ** 30:.1f} GB)", "",
         "| split folder | frames | videos | compressed GB |", "|---|---|---|---|"]
    for s, d in o["splits"].items():
        L.append(f"| {s} | {d['frames']} | {d['videos']} | {d['compressed_gb']:.1f} |")
    L += ["", f"Non-image files in zip: {json.dumps(o['non_image_files'])}", ""]
    t = o["splits"].get("test")
    if t:
        L += ["TEST frames per video: " + json.dumps(t["frames_per_video"]), ""]
        p = o["test_export_projection_h"]
        L += [f"Projected TEST export: {p['tiled_convnext_b_17_tiles_per_seed']:.1f} h per G5 seed (17 tiles), "
              f"{p['r0_plain_once']:.1f} h for R0 plain once; total {p['three_seeds_plus_r0']:.1f} h", ""]
    else:
        L += ["**No 'test' folder found in the zip.** TEST images must come from elsewhere; send this report to Claude.", ""]
    L += ["JSON files in the MARIVISION root (names, sizes and SHA only; none was opened as JSON):", ""]
    for j in o["json_files_in_root (names only, never parsed)"]:
        L.append(f"- {j['name']}  {j['bytes'] / 2 ** 20:.1f} MB  sha256 {j['sha256']}")
    return "\n".join(L) + "\n"


def selftest():
    import io
    import tempfile
    print(f"[selftest] {VERSION}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        zp = Path(td) / "SeaClips.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for split, nv, nf in (("train", 3, 5), ("val", 2, 4), ("test", 4, 6)):
                for v in range(nv):
                    for f in range(nf):
                        z.writestr(f"SeaClips/{split}/video_{v + 1:02d}/{f + 1:06d}.png", b"x" * 10)
            z.writestr("SeaClips/README.txt", b"r")
            z.writestr("SeaClips/test/", b"")
        sentinel = Path(td) / "seaclips-test.json"
        sentinel.write_text("THIS IS NOT JSON -- parsing it would raise", encoding="utf-8")
        o = inventory(zp, td)
        assert o["splits"]["train"]["frames"] == 15 and o["splits"]["val"]["frames"] == 8
        assert o["splits"]["test"] == {"frames": 24, "videos": 4, "frames_per_video": {f"video_{v:02d}": 6 for v in range(1, 5)},
                                       "compressed_gb": o["splits"]["test"]["compressed_gb"]}
        assert o["non_image_files"] == {".txt": 1}
        js = o["json_files_in_root (names only, never parsed)"]
        assert [j["name"] for j in js] == ["seaclips-test.json"] and len(js[0]["sha256"]) == 64
        assert abs(o["test_export_projection_h"]["tiled_convnext_b_17_tiles_per_seed"] - 24 * SEC_TILED_CNB / 3600) < 1e-12
        report(o)
    print("  [ok] frames per split / video from the zip list; non-images separated; a non-JSON label file is hashed, "
          "never parsed; export projection arithmetic")
    print("SELFTEST PASS")


def run():
    if (OUT / "g5_test_inventory.md").exists():
        sys.exit(f"[REFUSED] {OUT / 'g5_test_inventory.md'} exists")
    o = inventory(ZIP, ROOT)
    o["version"] = VERSION
    OUT.mkdir(parents=True, exist_ok=True)
    md = report(o)
    (OUT / "g5_test_inventory.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    (OUT / "g5_test_inventory.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"  wrote {OUT / 'g5_test_inventory.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run"], required=True)
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run()
