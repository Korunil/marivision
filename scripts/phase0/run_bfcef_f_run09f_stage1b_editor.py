
from __future__ import annotations
"""RUN_09F Stage 1B: one bounded, Train-only collaborative Editor pilot.
Requires unchanged RUN_09F Stage1 driver and certified RUN09C/RUN09F Stage0.
No Validation/Test. Folds 1-3 train, fold 4 calibrates, fold 0 evaluates once.
"""
import argparse
import gc
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import run_bfcef_f_run09f_stage1_editor_pilot as prior

SEED = 20261008
TASKS = ("cls", "box")
MODES = ("ProposerEditor", "CollaborativeEditor")
FROZEN_FRAME_BUDGET = 48
FROZEN_PROBES = 24
EPOCHS = 20
THRESHOLDS = (0.50, 0.65, 0.80)
NEW_DIR = prior.S0 / "run09f_editor_stage1b_v1"
OLD_REPORT = prior.OUT / "run09f_stage1_editor_pilot.json"
OLD_REPORT_SHA = "e6543534a69fe9d9422b54160ce98d37e2d10f060b9592084cc1ab16fa12b18c"


def consequence(full, reverted):
    """Exact global count utility of keeping Full rather than reverting.

    GT-identity changes are reported separately, not counted as net TP gains.
    """
    dt = int(full["tp"] - reverted["tp"])
    df = int(full["fp"] - reverted["fp"])
    gain = 4 * dt - df
    sf, sr = set(full["matched_gt_indices"]), set(reverted["matched_gt_indices"])
    ff, fr = set(full["fp_dict"]), set(reverted["fp_dict"])
    identity = (len(sf - sr), len(sr - sf), len(fr - ff), len(ff - fr))
    # 0=harmful to keep, 1=neutral, 2=beneficial to keep.
    label = 0 if gain < 0 else (2 if gain > 0 else 1)
    return gain, label, identity


def reviewed_features(z0, b0, z1, b1, oof, info, pi_c, pi_b, q, c):
    p, r = prior.features(z0, b0, z1, b1, oof, pi_c, pi_b, q, c)
    task = "cls" if c is not None else "box"
    raw = info.get("cls_channels" if task == "cls" else "box_channels", {})
    for e in prior.EXPERTS:
        channel = raw.get(e, {})
        for key in ("U", "H", "sigma"):
            t = channel.get(key)
            val = 0.0 if t is None else float(t[q].detach().reshape(-1)[0])
            p.append(val)
    assert len(p) == (28 if task == "cls" else 23)
    assert len(r) == 8
    if not all(math.isfinite(x) for x in p + r):
        raise AssertionError("Nonfinite inference-available editorial features")
    return p, r


def eligible_slots(z0, b0, z1, b1, oof, quota):
    """Identical GT-free proposal policy for training, calibration and inference."""
    base = prior.r8.extract_top100_predictions(z0, b0)
    full = prior.r8.extract_top100_predictions(z1, b1)
    pairs = set(zip(base["all_q_idx"].tolist(), base["all_c_idx"].tolist()))
    pairs.update(zip(full["all_q_idx"].tolist(), full["all_c_idx"].tolist()))
    queries = sorted({int(q) for q in oof["candidate_queries"].tolist()})
    result = {"cls": [], "box": []}
    for task in TASKS:
        slots = prior.sample_indices(z0, b0, z1, b1, queries, task, quota)
        for q, c in slots:
            if task == "cls" and (q, c) in pairs and abs(float(z1[q,c]-z0[q,c])) > 1e-7:
                result[task].append((q, c))
            elif task == "box" and any(q == qq for qq, _ in pairs) and float((b1[q]-b0[q]).abs().max()) > 1e-7:
                result[task].append((q, None))
    return result


def frame_rows(fr, oof, z0, b0, z1, b1, info, fold, video):
    pi_c = prior.pi_or_baseline(info, "pi_cls")
    pi_b = prior.pi_or_baseline(info, "pi_box")
    slots = eligible_slots(z0, b0, z1, b1, oof, FROZEN_PROBES)
    _, full = prior.match(z1, b1, fr)
    result = {t: [] for t in TASKS}
    for task in TASKS:
        for q, c in slots[task]:
            zr, br = z1.clone(), b1.clone()
            if task == "cls":
                zr[q,c] = z0[q,c]
                assert torch.equal(zr[q,c], z0[q,c])
            else:
                br[q] = b0[q]
                assert torch.equal(br[q], b0[q])
            _, reversed_match = prior.match(zr, br, fr)
            gain, label, identities = consequence(full, reversed_match)
            p, r = reviewed_features(z0, b0, z1, b1, oof, info, pi_c, pi_b, q, c)
            result[task].append({
                "p": p, "r": r, "y": label, "gain": gain,
                "video": int(video), "fold": int(fold),
                "gt_identity": identities,
            })
    return result


class DecisionEditor(nn.Module):
    """Three-way harm/neutral/benefit adjudicator; reviewer votes learned end-to-end."""
    def __init__(self, task, collaborative):
        super().__init__()
        self.collaborative = collaborative
        self.pdim = 28 if task == "cls" else 23
        self.register_buffer("pm", torch.zeros(self.pdim))
        self.register_buffer("ps", torch.ones(self.pdim))
        self.register_buffer("rm", torch.zeros(8))
        self.register_buffer("rs", torch.ones(8))
        if collaborative:
            self.reviewers = nn.ModuleList([
                nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3))
                for _ in range(4)
            ])
        self.editor = nn.Sequential(
            nn.Linear(self.pdim + (12 if collaborative else 0), 40),
            nn.ReLU(), nn.Linear(40, 24), nn.ReLU(), nn.Linear(24, 3)
        )

    def forward(self, p, r):
        if self.collaborative:
            # Each reviewer receives only that expert's two proposal-evidence features.
            votes = [
                self.reviewers[i](r[:, [i, i+4]]).softmax(-1)
                for i in range(4)
            ]
            p = torch.cat([p, torch.cat(votes, dim=-1)], dim=-1)
        return self.editor(p)


def tensors(rows):
    p = torch.tensor([x["p"] for x in rows], dtype=torch.float32)
    r = torch.tensor([x["r"] for x in rows], dtype=torch.float32)
    y = torch.tensor([x["y"] for x in rows], dtype=torch.long)
    return p, r, y


def train_editor(task, collab, rows):
    train = [x for x in rows if x["fold"] in (1,2,3)]
    classes = Counter(x["y"] for x in train)
    bad_vids = {x["video"] for x in train if x["y"] == 0}
    support = {"n": len(train), "harm": classes[0], "neutral": classes[1],
               "benefit": classes[2], "harm_video_count": len(bad_vids)}
    if len(train) < 200 or classes[0] < 15 or len(bad_vids) < 3 or classes[2] < 15:
        print(f"[BLOCK TASK] {task} lacks diverse harmful/beneficial support: {support}")
        support["bypass"] = True
        return None, support
    torch.manual_seed(SEED + 37 * (task == "box") + int(collab))
    net = DecisionEditor(task, collab)
    p, r, y = tensors(train)
    pm, ps = p.mean(0), p.std(0, unbiased=False).clamp_min(.05)
    # One-hot class ID does not need centering or scaling.
    if task == "cls": pm[8:15] = 0; ps[8:15] = 1
    rm, rs = r.mean(0), r.std(0, unbiased=False).clamp_min(.05)
    net.pm.copy_(pm); net.ps.copy_(ps); net.rm.copy_(rm); net.rs.copy_(rs)
    p, r = (p-pm)/ps, (r-rm)/rs
    weights = torch.tensor([min(6., math.sqrt(len(train)/max(classes[i], 1))) for i in range(3)])
    weights = weights / weights.mean()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED+epoch))
        net.train()
        for batch in order.split(512):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(p[batch], r[batch]), y[batch], weight=weights)
            if not torch.isfinite(loss): raise AssertionError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    support["bypass"] = False
    support["weight"] = weights.tolist()
    return net, support


def predictions(net, rows):
    if net is None or not rows: return torch.empty((0,3))
    p,r,_ = tensors(rows)
    with torch.no_grad():
        out = net((p-net.pm)/net.ps,(r-net.rm)/net.rs).softmax(-1)
    if not torch.isfinite(out).all(): raise AssertionError("nonfinite editorial probabilities")
    return out


def calibrate(net, rows):
    """One predeclared Train fold-4 calibration; approve-all if no net evidence."""
    val = [x for x in rows if x["fold"] == 4]
    if net is None: return None, {"reason": "insufficient training support", "n": len(val)}
    if not val: return None, {"reason": "empty calibration", "n": 0}
    pred = predictions(net, val)
    gains = np.asarray([x["gain"] for x in val], dtype=np.float64)
    best = None
    table = []
    for t in THRESHOLDS:
        # Deny only when harmful is more likely than beneficial, and confidence is high.
        deny = ((pred[:,0] >= t) & (pred[:,0] > pred[:,2])).numpy()
        saved = -float(gains[deny].sum())
        vids = len({val[i]["video"] for i in np.flatnonzero(deny)})
        row = {"threshold":t, "reject":int(deny.sum()), "reject_videos":vids,
               "counterfactual_utility_saved":saved,
               "false_rejects":int(sum(val[i]["gain"]>0 for i in np.flatnonzero(deny))),
               "true_rejects":int(sum(val[i]["gain"]<0 for i in np.flatnonzero(deny)))}
        table.append(row)
        # Do not activate a rule justified only by one video or one event.
        if saved > 0 and row["reject"] >= 2 and vids >= 2:
            if best is None or (saved, -row["reject"]) > (best["counterfactual_utility_saved"],-best["reject"]):
                best = row
    return (None if best is None else float(best["threshold"])), {
        "n": len(val), "class_counts": dict(Counter(x["y"] for x in val)),
        "candidate_thresholds": table, "selected": best,
        "note": "sampled single-revert Fold4 utility; not a joint global performance claim"
    }


def collect_train_and_calibration(model, CO, FX, folds):
    rows = {t: [] for t in TASKS}
    counts = Counter()
    with torch.no_grad():
        for fold in (1,2,3,4):
            prior.r9.load_source_fold(model, fold)
            for video in folds[fold]:
                can, spa, tmp, _ = prior.r9.load_video_all(video, fold)
                oof = torch.load(prior.r9.s0a.fold_oof_path(fold, video), map_location="cpu", weights_only=False)
                assert len(can) == len(oof)
                ix = sorted(set(np.linspace(0, len(can)-1, min(len(can),FROZEN_FRAME_BUDGET),dtype=int).tolist()))
                for i in ix:
                    O,X = prior.r9.s0a.make_O_X(can[i], spa[i], tmp[i], CO, FX)
                    z1,b1,info = model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode="auto")
                    z0,b0 = O.z.detach().cpu().float(),O.b.detach().cpu().float()
                    zz,bb = z1.detach().cpu().float(),b1.detach().cpu().float()
                    frame = frame_rows(can[i], oof[i], z0,b0,zz,bb,info,fold,video)
                    for task in TASKS:
                        rows[task].extend(frame[task]);counts[f"f{fold}_{task}"] += len(frame[task])
                    del O,X,z1,b1,info
                print(f"[+] fold={fold} video={video:02d} frames={len(ix)} "
                      f"cls={counts[f'f{fold}_cls']} box={counts[f'f{fold}_box']}")
                del can,spa,tmp,oof
                gc.collect()
    return rows,dict(counts)


def editorial_apply(z0,b0,z1,b1,oof,info,editors,threshold=None):
    """Called by certified Fold0 evaluator; reject = exact baseline value."""
    z,b = z1.clone(),b1.clone()
    pi_c = prior.pi_or_baseline(info,"pi_cls")
    pi_b = prior.pi_or_baseline(info,"pi_box")
    slots = eligible_slots(z0,b0,z1,b1,oof,FROZEN_PROBES)
    result = Counter()
    for task in TASKS:
        net,t = editors[task]
        if net is None or t is None: continue
        row = []
        for q,c in slots[task]:
            p,r = reviewed_features(z0,b0,z1,b1,oof,info,pi_c,pi_b,q,c)
            row.append({"p":p,"r":r,"y":1})
        if not row: continue
        pred = predictions(net,row)
        deny = (pred[:,0]>=t) & (pred[:,0]>pred[:,2])
        result[f"{task}_eligible"] += len(row)
        for (q,c),rejection in zip(slots[task],deny.tolist()):
            if not rejection: continue
            if task == "cls":
                z[q,c] = z0[q,c]
                assert torch.equal(z[q,c],z0[q,c])
            else:
                b[q] = b0[q]
                assert torch.equal(b[q],b0[q])
            result[f"{task}_reject"] += 1
    return z,b,dict(result)


def selftest():
    a={"tp":2,"fp":3,"matched_gt_indices":{1,2},"fp_dict":{(1,2):None}}
    b={"tp":1,"fp":5,"matched_gt_indices":{1},"fp_dict":{(2,1):None}}
    assert consequence(a,b)[:2] == (6,2)
    x=DecisionEditor("cls",True)
    assert x(torch.randn(3,28),torch.randn(3,8)).shape == (3,3)
    z=torch.zeros(300,7); zz=z.clone(); zz[0,0]=1
    bx=torch.rand(300,4); bb=bx.clone(); bb[0,0]+=0.1
    f=prior.features(z,bx,zz,bb,{"expert_cls_logits":torch.zeros(4,300,7),
      "expert_box_boxes":torch.zeros(4,300,4)},torch.full((300,5),.2),torch.full((300,5),.2),0,0)
    assert len(f[0])==16 and len(f[1])==8
    assert consequence(a,a)[:2] == (0,1)
    # A changed GT identity with unchanged total TP must not generate net TP credit.
    swap_a={"tp":1,"fp":2,"matched_gt_indices":{1},"fp_dict":{}}
    swap_b={"tp":1,"fp":2,"matched_gt_indices":{2},"fp_dict":{}}
    assert consequence(swap_a,swap_b)[0] == 0
    p,r = reviewed_features(z,bx,zz,bb,
       {"expert_cls_logits":torch.zeros(4,300,7),"expert_box_boxes":torch.zeros(4,300,4)},
       {},torch.full((300,5),.2),torch.full((300,5),.2),0,0)
    assert len(p)==28 and len(r)==8
    print("[PASS] Stage1B synthetic global-count/editor/feature tests")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--self-test-only", action="store_true")
    args=ap.parse_args()
    selftest()
    if args.self_test_only:return
    prior.gate(OLD_REPORT,OLD_REPORT_SHA)
    prior.read_only_inputs()
    reference=json.loads(OLD_REPORT.read_text())
    assert reference["editor_interventions"]["CollaborativeEditor"]["cls_reject"]==0
    folds=prior.r9.parse_folds()
    assert sorted(folds[0])==[6,7,14,16,17,24,36,37,40,42]
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    model,CO,FX=prior.initialize()
    rows,support_count=collect_train_and_calibration(model,CO,FX,folds)
    support={};decisions={};nets={}
    for mode in MODES:
        decisions[mode]={};support[mode]={};nets[mode]={}
        for task in TASKS:
            net,ss=train_editor(task,mode=="CollaborativeEditor",rows[task])
            t,cal=calibrate(net,rows[task])
            decisions[mode][task]=t
            nets[mode][task]=(net,t)
            support[mode][task]={"train":ss,"fold4_calibration":cal}
            print(f"[+] {mode}/{task} editor threshold={t} train={ss}")
    # The Editor approach fails if neither task is supported by any variant.
    if not any(decisions[m][t] is not None for m in MODES for t in TASKS):
        print("[NO-GO] No reliable Train-only editorial headroom; Fold0 not re-opened.")
        result={"status":"NO_GO_EDITOR","stage":"pre-fold0","support":support,"counts":support_count}
    else:
        # Both variants use the same GT-free eligible-slot and veto routine.
        # Reuse exact certified COCO Fold0 evaluator; pass settings by closure.
        original=prior.apply_editors
        prior.apply_editors=editorial_apply
        try:
            metrics, interventions=prior.evaluate_fold0(model,CO,FX,folds,nets,threshold=None)
        finally:
            prior.apply_editors=original
        result={"status":"FOLD0_DEVELOPMENT_COMPARISON","metrics":metrics,"interventions":interventions,
                "support":support,"counts":support_count}
        print("="*106)
        for v,m in metrics.items():
            print(f"{v:<21} mAP={m['map30']:.5f} mAPca={m['mapca30']:.5f} "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']} F1={m['f1']:.5f}")
        print("[+] editor interventions:",interventions)
    NEW_DIR.mkdir(parents=True,exist_ok=True)
    paths={}
    for mode in MODES:
        p=NEW_DIR/(mode.lower()+"_stage1b.pth")
        torch.save({"schema":"run09f_stage1b_v1","source_ckpt_sha":prior.CKPT_SHA,
                    "folds_train":[1,2,3],"fold_calibration":4,"fold_development":0,
                    "thresholds":decisions[mode],
                    "editor_state":{t:(nets[mode][t][0].state_dict() if nets[mode][t][0] else None)
                                    for t in TASKS}},p)
        paths[mode]={"path":p.name,"sha256":prior.sha(p)}
    result["_meta"]={"schema":"bfcef_f_run09f_stage1b_v1","train_only":True,"test_embargo":True,
         "old_report_sha256":OLD_REPORT_SHA,"frozen_source_checkpoint_sha256":prior.CKPT_SHA,
         "source_script_sha256":prior.sha(Path(__file__)),"sample_frames_per_video":FROZEN_FRAME_BUDGET,
         "probes_per_task_per_frame":FROZEN_PROBES,"epochs":EPOCHS,"threshold_candidates":THRESHOLDS,
         "limitations":"Fold0 previously used in development; base R50 trained on Train; calibrated on Train fold4; single-action global supervision, joint veto outcome evaluated only."}
    result["checkpoint_paths"]=paths
    report=NEW_DIR/"run09f_stage1b_report.json"
    report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("[+] Report:",report,"SHA256:",prior.sha(report))
    print("[PASS] Stage1B complete. STOP: no model changes, Validation or Test.")


if __name__=="__main__":main()
