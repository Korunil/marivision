"""
r0_preflight.py (v4)
EXP-R0-PREFLIGHT-v4: Authoritative Model, Checkpoint, and Transform Audit
Features:
  - Official parser inheritance (get_args_parser from main.py)
  - Pretrained COCO checkpoint architecture assertions
  - Clean 7-class head with shared weights across decoder stages
  - Optimizer partition completeness (P_core, P_backbone, P_linear disjoint union)
  - Real-data transform verification using datasets/seaclips_detection.py
  - Forward-backward smoke test on batch of 2 images
Run: python r0_preflight.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image

from main import get_args_parser
from models.deformable_detr import build
from datasets.seaclips_detection import SeaClipsDetection, IMAGENET_MEAN, IMAGENET_STD

COCO_CKPT_PATH = Path("D:/MARIVISION/r50_deformable_detr-checkpoint.pth")
TRAIN_JSON_PATH = Path("D:/MARIVISION/seaclips-train.json")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

def match_name_keywords(n, name_keywords):
    for b in name_keywords:
        if b in n:
            return True
    return False

def get_authoritative_r0_args():
    parser = get_args_parser()
    args = parser.parse_args([])

    # Explicit Supplementary Table 2(b) and Section B overrides
    args.lr = 2e-4
    args.lr_backbone = 2e-5
    args.lr_linear_proj_mult = 0.1
    args.weight_decay = 1e-4
    args.epochs = 15
    args.batch_size = 2
    args.clip_max_norm = 0.1

    args.num_queries = 300
    args.num_feature_levels = 4
    args.aux_loss = True
    args.two_stage = False
    args.with_box_refine = False
    args.dilation = False

    args.dataset_file = 'seaclips'
    args.num_classes_override = 7
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def inspect_and_assert_checkpoint(ckpt_path, target_args):
    print("\n[1] Auditing Pretrained COCO Checkpoint Architecture...")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'args' in ckpt:
        c_args = ckpt['args']
        print(f"    Checkpoint args: two_stage={getattr(c_args, 'two_stage', 'N/A')}, "
              f"with_box_refine={getattr(c_args, 'with_box_refine', 'N/A')}, "
              f"dilation={getattr(c_args, 'dilation', 'N/A')}")
        assert getattr(c_args, 'num_queries', 300) == target_args.num_queries
        assert getattr(c_args, 'num_feature_levels', 4) == target_args.num_feature_levels
        assert getattr(c_args, 'two_stage', False) == target_args.two_stage
        assert getattr(c_args, 'with_box_refine', False) == target_args.with_box_refine
        assert getattr(c_args, 'dilation', False) == target_args.dilation
        print("    [+] Checkpoint architecture MATCHES target R0 specification.")

    state = ckpt['model'] if 'model' in ckpt else ckpt
    assert state['query_embed.weight'].shape == (300, 512), "query_embed shape mismatch"
    assert state['transformer.level_embed'].shape == (4, 256), "level_embed shape mismatch"
    return state

def run_preflight():
    print("=" * 105)
    print("EXP-R0-PREFLIGHT-v4: AUTHORITATIVE PREFLIGHT INVARIANT AUDIT")
    print("=" * 105)

    args = get_authoritative_r0_args()
    raw_state = inspect_and_assert_checkpoint(COCO_CKPT_PATH, args)

    print("\n[2] Building Model via Official Minimal-Diff Builder...")
    model, criterion, postproc = build(args)

    # Class Head Assertions
    assert len(model.class_embed) == 6, f"Expected 6 decoder heads, got {len(model.class_embed)}"
    assert all(h.out_features == 7 for h in model.class_embed), "Class head out_features != 7"
    assert all(h.in_features == 256 for h in model.class_embed), "Class head in_features != 256"
    assert all(model.class_embed[i].weight is model.class_embed[0].weight for i in range(1, 6)), \
        "Classifier heads are not shared across decoder stages!"
    assert criterion.num_classes == 7, "Criterion num_classes != 7"
    print("    [+] MODEL CLASSES: 7 | CRITERION CLASSES: 7 | SHARED HEAD: Verified (PASS)")

    # Checkpoint Transfer
    print("\n[3] Auditing Clean Classifier Transfer...")
    filtered_state = {k: v for k, v in raw_state.items() if not k.startswith("class_embed.")}
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    assert len(unexpected) == 0, f"Unexpected checkpoint keys: {unexpected}"
    assert all(k.startswith("class_embed.") for k in missing), f"Non-classifier missing keys: {missing}"
    print(f"    [+] CHECKPOINT TRANSFER: PASS (Retained 100% backbone/transformer; skipped {len(missing)} class weights)")

    # Backbone Freezing Policy Audit
    print("\n[4] Auditing Official Backbone Freezing Policy...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    expected_frozen, unexpected_frozen = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if "backbone.0.body" in name and any(x in name for x in ["conv1", "bn1", "layer1"]):
                expected_frozen.append(name)
            else:
                unexpected_frozen.append(name)

    print(f"    Total Parameters:            {total_params:,}")
    print(f"    Trainable Parameters:        {trainable_params:,}")
    print(f"    Official-Backbone-Frozen:    {frozen_params:,} ({len(expected_frozen)} tensors in stem/layer1)")
    print(f"    Unexpected Frozen:           {len(unexpected_frozen)}")
    assert len(unexpected_frozen) == 0, f"Unexpected frozen parameters: {unexpected_frozen}"
    print("    [+] FREEZING POLICY: PASS (Stem/layer1 frozen per official implementation; all others trainable)")

    # Optimizer Group Partition Completeness
    print("\n[5] Auditing Official 3-Group Parameter Partition...")
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    s_core = set(id(p) for p in param_dicts[0]["params"])
    s_backbone = set(id(p) for p in param_dicts[1]["params"])
    s_linear = set(id(p) for p in param_dicts[2]["params"])
    s_trainable = set(id(p) for p in model.parameters() if p.requires_grad)

    assert s_core.isdisjoint(s_backbone), "Overlap: Core and Backbone!"
    assert s_core.isdisjoint(s_linear), "Overlap: Core and Linear Proj!"
    assert s_backbone.isdisjoint(s_linear), "Overlap: Backbone and Linear Proj!"
    assert (s_core | s_backbone | s_linear) == s_trainable, "Partition incomplete!"

    print(f"    Group 1 (Core):       {len(s_core)} tensors | LR = {param_dicts[0]['lr']:.1e}")
    print(f"    Group 2 (Backbone):   {len(s_backbone)} tensors | LR = {param_dicts[1]['lr']:.1e}")
    print(f"    Group 3 (LinearProj): {len(s_linear)} tensors | LR = {param_dicts[2]['lr']:.1e}")
    print("    [+] OPTIMIZER PARTITION: Mutually disjoint union complete (PASS)")

    # Real-Data Transform Equivalence Audit
    print("\n[6] Auditing Real-Data Preprocessing Pipeline...")
    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON_PATH, is_train=False)
    sample_tensor, sample_target = dataset[0]

    assert sample_tensor.shape == (3, 640, 640), f"Sample tensor shape mismatch: {sample_tensor.shape}"
    if len(sample_target['boxes']) > 0:
        assert (sample_target['boxes'] >= 0.0).all() and (sample_target['boxes'] <= 1.0).all(), "Boxes not in [0, 1]"
        assert (sample_target['labels'] >= 0).all() and (sample_target['labels'] <= 6).all(), "Labels not in [0, 6]"
        assert (sample_target['boxes'][:, 2:] > 0.0).all(), "Zero-width/height boxes present"

    # Independent reference comparison on un-normalized image
    img_info = dataset.images[dataset.img_ids[0]]
    raw_pil = dataset._load_image(img_info)
    import torchvision.transforms.functional as TF
    raw_resized_t = TF.to_tensor(TF.resize(raw_pil, [640, 640]))
    mean_t = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std_t = torch.tensor(IMAGENET_STD)[:, None, None]
    expected_norm = (raw_resized_t - mean_t) / std_t

    max_err = (sample_tensor - expected_norm).abs().max().item()
    print(f"    Real-Data Normalization Discrepancy: {max_err:.2e}")
    assert max_err < 1e-6, "Real-data normalization arithmetic error!"
    print("    [+] REAL-DATA PREPROCESSING: Bitwise verification PASS")

    # Forward-Backward Smoke Test
    print("\n[7] Running Forward-Backward Smoke Test (Batch Size = 2)...")
    device = torch.device(args.device)
    model.to(device)
    criterion.to(device)

    batch_tensors = torch.stack([dataset[0][0], dataset[1][0]], dim=0).to(device)
    batch_targets = [{k: v.to(device) for k, v in dataset[i][1].items()} for i in [0, 1]]

    from util.misc import NestedTensor
    nested_inputs = NestedTensor(batch_tensors, torch.zeros(2, 640, 640, dtype=torch.bool, device=device))

    outputs = model(nested_inputs)
    loss_dict = criterion(outputs, batch_targets)
    losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

    model.zero_grad()
    losses.backward()

    # Check finite gradients on trainable parameters and zero on frozen
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient on trainable parameter: {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient on: {name}"
        else:
            assert p.grad is None, f"Gradient accumulated on frozen parameter: {name}"

    print(f"    Total Loss: {losses.item():.4f} | Backward Pass: Finite gradients verified.")
    print("    [+] FORWARD-BACKWARD SMOKE TEST: PASS")

    print("\n" + "=" * 105)
    print("EXP-R0-PREFLIGHT-v4 VERDICT: ALL INVARIANTS SATISFIED (PASS)")
    print("=" * 105)

if __name__ == "__main__":
    run_preflight()