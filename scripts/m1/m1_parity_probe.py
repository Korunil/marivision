r"""m1_parity_probe.py - READ-ONLY diagnostic for an M1 provenance miss (MARIVISION-HR-P2, 2026-10-27). Writes nothing.

For each requested VAL image it runs the frozen R0 full frame through several paths and prints, side by side, every
detection with score >= 0.05 from:
  REF      the stored plain export (--reference-plain-json)
  EXPORT   the plain exporter's exact path: dataset[idx] tensors, batch of 2 with the same batch partner the exporter
           used (DataLoader order, batch_size 2), util.misc.collate_fn
  M1_b1    this M1 preprocessing (PIL crop of full frame -> 640x640), batch of 1
  M1_b4    M1 preprocessing, batch of 4 (the frame + 3 copies of the next frames), as in the failed run
It also reports the max |tensor difference| between the dataset tensor and the M1 tensor.
Interpretation: if EXPORT != REF the stored file itself is not reproducible (GPU nondeterminism at export time);
if M1_b1 == EXPORT but M1_b4 differs, it is batch-size numerical sensitivity; if the tensors differ, it is preprocessing.

Run from D:\MARIVISION\R0_frozen_baseline:
  set R0_MSDA_BACKEND=pytorch
  python m1_parity_probe.py --checkpoint checkpoint_best_r0_epoch04.pth ^
      --reference-plain-json val_exports\r0_val_predictions_epoch04.json --image-ids 3776
"""
from PIL import Image  # noqa: F401
import argparse
import json
import os
import sys
from pathlib import Path

STRICT = 0.05


def fmt(d):
    x, y, w, h = d["bbox"]
    return f"cls {d['category_id']} s={d['score']:.5f} box=({x:8.2f},{y:8.2f},{w:7.2f},{h:7.2f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference-plain-json", required=True)
    ap.add_argument("--image-ids", required=True, help="Comma-separated VAL image ids, e.g. 3776")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--export-batch-size", type=int, default=2, help="Batch size the plain export used (default 2)")
    a = ap.parse_args()
    if os.environ.get("R0_MSDA_BACKEND", "").lower() != "pytorch":
        raise SystemExit("[REFUSED] Run 'set R0_MSDA_BACKEND=pytorch' first.")
    if "test" in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST is locked.")

    import torch
    import torchvision.transforms.functional as TF
    sys.path.insert(0, str(Path.cwd()))
    from models.deformable_detr import build
    from datasets.seaclips_detection import SeaClipsDetection
    from r0_preflight import get_authoritative_r0_args
    from util.misc import NestedTensor, collate_fn

    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    ref = json.loads(Path(a.reference_plain_json).read_text(encoding="utf-8"))
    try:
        ckpt = torch.load(Path(a.checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(Path(a.checkpoint), map_location="cpu")
    args = get_authoritative_r0_args()
    device = torch.device(args.device)
    model, _c, pp = build(args)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    ds = SeaClipsDetection(Path(a.zip), Path(a.coco_val), is_train=False, random_flip=False)
    l2c = {int(k): int(v) for k, v in dict(ds.label_to_cat_id).items()}
    order = list(ds.img_ids)

    def m1_tensor(iid):
        pil = ds._load_image(ds.images[iid]).convert("RGB")
        W, H = pil.size
        t = TF.to_tensor(TF.resize(pil.crop((0, 0, W, H)), [640, 640]))
        return (t - mean) / std, (H, W)

    def dets_from(res, iid):
        out = []
        for s, lb, (x1, y1, x2, y2) in zip(res["scores"].cpu().tolist(), res["labels"].cpu().tolist(), res["boxes"].cpu().tolist()):
            out.append({"category_id": l2c[int(lb)], "score": float(s), "bbox": [x1, y1, x2 - x1, y2 - y1]})
        return out

    with torch.inference_mode():
        for iid in [int(x) for x in a.image_ids.split(",")]:
            k = order.index(iid)
            b0 = (k // a.export_batch_size) * a.export_batch_size
            batch_idx = list(range(b0, min(len(order), b0 + a.export_batch_size)))
            print(f"\n================ image {iid} (dataset index {k}); exporter batch indices {batch_idx} "
                  f"-> image ids {[order[i] for i in batch_idx]}")
            samples, targets = collate_fn([ds[i] for i in batch_idx])
            samples = samples.to(device)
            res = pp["bbox"](model(samples), torch.stack([t["orig_size"] for t in targets]).to(device))
            paths = {"EXPORT": dets_from(res[batch_idx.index(k)], iid)}
            t_m1, (H, W) = m1_tensor(iid)
            t_ds = ds[k][0]
            print(f"  max |dataset tensor - M1 tensor| = {(t_ds - t_m1).abs().max().item():.3e}   orig_size(dataset)="
                  f"{targets[batch_idx.index(k)]['orig_size'].tolist()}  M1 (h,w)=({H},{W})")
            for bs in (1, 4):
                ids = [iid] + [order[min(len(order) - 1, k + j)] for j in range(1, bs)]
                ts = [m1_tensor(i) for i in ids]
                x = torch.stack([t for t, _ in ts]).to(device)
                out = model(NestedTensor(x, torch.zeros(x.shape[0], 640, 640, dtype=torch.bool, device=device)))
                r = pp["bbox"](out, torch.tensor([[h, w] for _, (h, w) in ts], dtype=torch.float32, device=device))
                paths[f"M1_b{bs}"] = dets_from(r[0], iid)
            paths["REF"] = [d for d in ref if int(d["image_id"]) == iid]
            for name in ("REF", "EXPORT", "M1_b1", "M1_b4"):
                strict = sorted([d for d in paths[name] if d["score"] >= STRICT], key=lambda d: -d["score"])
                print(f"  --- {name}: {len(strict)} dets >= {STRICT}")
                for d in strict:
                    print("     " + fmt(d))
            # near-threshold region, where rank/score churn would show
            for name in ("REF", "M1_b4"):
                near = sorted([d for d in paths[name] if 0.03 <= d["score"] < STRICT], key=lambda d: -d["score"])
                print(f"  --- {name}: {len(near)} dets in [0.03, 0.05): " + "; ".join(fmt(d) for d in near[:6]))


if __name__ == "__main__":
    main()
