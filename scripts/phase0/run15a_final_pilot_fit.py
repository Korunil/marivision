"""Fresh downstream short FIT-only pilot; NEVER a full R50 retrain.

No evaluation/calibration/test read; source-pinned frozen R0 and full experts.
Pilot deliberately uses few FIT frames and makes no qualification claims.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random
from pathlib import Path
from PIL import Image
import torch
from run15a_final_mechanism import mechanism_forward
from run15a_final_risk import RiskPolicy
from run15a_final_preservation import protected_gt_loss,identity_ledger
from run15a_final_supervision import isolated_native_targets,expert_deep_supervision
from run15a_final_windows import load_runtime, require,write_new,sha,FIT


def pilot(project,vids,max_frames):
    require(max_frames>=1 and max_frames<=64,'Pilot budget max 64 frames/video')
    require(len(vids)>0 and len(vids)<=3 and all(v in FIT for v in vids),'Pilot only <=3 FIT videos')
    old,r9,r8,d,s0a,raw,unused_model,CO,FX=load_runtime(project, downstream="none")
    # New U/H/TSK/router, frozen full-Train experts; V2 downstream NOT reused.
    r12=old.imports_and_verify()[1]
    model,criterion,CO,FX=old.make_model(r9,r12)
    r9.set_training_modes(model,joint=True)
    old.lock_gate_thresholds(model)
    params=[p for p in model.parameters() if p.requires_grad]
    require(bool(params),'No downstream trainable parameters')
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    old.verify_gate_thresholds(model,'final pilot before steps',optimizer)
    model.train();model.expert_s.eval();model.expert_l.eval();model.expert_c.eval();model.expert_t.eval();model.adapter.eval();model.context_bus.eval()
    records=[];seen=0
    for vid in vids:
        can,spa,tmp=s0a.load_video_triplet(vid)
        for j in range(min(max_frames,len(can))):
            fr=can[j];O,X=s0a.make_O_X(fr,spa[j],tmp[j],CO,FX)
            bundle=r9.expert_bundle(model,O,X)
            matcher=old.capture_match
            targets=isolated_native_targets(model,O,bundle,fr['gt_boxes_norm'],fr['gt_labels'],matcher)
            optimizer.zero_grad(set_to_none=True)
            out=mechanism_forward(model,O,X,bundle,hard=False,policy=RiskPolicy())
            matched_base,_,_=matcher(O.z,O.b,fr['gt_boxes_norm'],fr['gt_labels'])
            preservation=protected_gt_loss(O.z,O.b,out['z'],out['b'],fr['gt_boxes_norm'].to(O.h.device),matched_base)
            deep,details=expert_deep_supervision(out,targets)
            detection=r9.detection_loss(criterion,out['z'],out['b'],fr)
            route=(out['pi_cls'][:,1:].square().mean()+out['pi_box'][:,1:].square().mean())
            loss=deep + 5.0*preservation['total']+0.20*detection+0.003*route
            require(bool(torch.isfinite(loss)), 'Nonfinite pilot objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,.1)
            optimizer.step()
            old.verify_gate_thresholds(model,f'final pilot video={vid} frame={j}',optimizer)
            with torch.no_grad():
                new=mechanism_forward(model,O,X,bundle,hard=True,policy=RiskPolicy())
                matched_new,_,_=matcher(new['z'],new['b'],fr['gt_boxes_norm'],fr['gt_labels'])
                ledger=identity_ledger(matched_base,matched_new)
            seen+=1;records.append({'video':vid,'frame':j,'loss':float(loss.detach()),
                 'deep':float(deep.detach()),'preservation':float(preservation['total'].detach()),
                 'baseline_identities_lost':len(ledger['lost']),'new_identities_rescued':len(ledger['rescued']),
                 'cls_gate_open':int((out['hard_cls'].sum(-1)>0).sum()),
                 'box_gate_open':int((out['hard_box'].sum(-1)>0).sum())})
            print(f'[PILOT] video={vid} frame={j} loss={records[-1]["loss"]:.5f}',flush=True)
    old.verify_gate_thresholds(model,'final pilot completed',optimizer)
    root=Path(project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'pilot_fit'
    root.mkdir(parents=True,exist_ok=True)
    filename=root/('pilot_videos_'+'_'.join(map(str,vids))+f'_frames{max_frames}.pth')
    require(not filename.exists(),'Preserve existing pilot checkpoint; refusing overwrite')
    state=old.state_downstream(model)
    payload={'schema':'run15a_final_candidate_pilot_v1','source_sha256':sha(Path(__file__)),
             'R0_frozen_sha256':old.PARENT_SHA,'video_ids':vids,'max_frames_per_video':max_frames,
             'checkpoint_type':'PILOT_ONLY_NOT_CALIBRATED','downstream_state':state}
    torch.save(payload,filename)
    recfile=filename.with_suffix('.receipt.json')
    write_new(recfile,{'status':'FIT_PILOT_ONLY_NOT_QUALIFIED','checkpoint_sha256':sha(filename),
                       'source_sha256':sha(Path(__file__)),'frames':seen,'records':records,
                       'CAL_QUAL_official_val_test_unopened':True,
                       'warnings':['GT 3x3 labels based on isolated global expert interventions; interaction attribution imperfect',
                                   'Fresh downstream pilot weights NOT qualified or calibrated',
                                   'Constrained surrogate does NOT guarantee no GT losses on unseen frames']})
    return recfile

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    p.add_argument('--videos',default='2');p.add_argument('--max-frames',type=int,default=4)
    a=p.parse_args();print('[PILOT COMPLETE]',pilot(a.project,tuple(map(int,a.videos.split(','))),a.max_frames))
