"""Independent synthetic/data-free adversarial tests for RUN15A PATCH02 recovery."""
import ast,importlib.util,hashlib
from pathlib import Path
p=Path(__file__).with_name('run_bfcef_f_run15a_qualification_recovery_PATCH02.py')
spec=importlib.util.spec_from_file_location('patch02_recovery_independent',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
N=0
def check(ok,message):
    global N
    assert ok,message
    N+=1
cat=[{'id':i,'name':f'c{i}'} for i in range(1,8)]
D={'images':[{'id':11,'video_id':4,'width':600,'height':400},
             {'id':12,'video_id':4,'width':600,'height':400},
             {'id':13,'video_id':5,'width':600,'height':400}],
   'videos':[{'id':4},{'id':5}],'categories':cat,
   'annotations':[{'id':11,'image_id':11,'category_id':2,'bbox':[0,0,12,8]},
                  {'id':12,'image_id':12,'category_id':5,'bbox':[0,0,12,8]},
                  {'id':13,'image_id':13,'category_id':1,'bbox':[0,0,12,8]}]}
R={'video':4,'image_id':11,'class_gt':[1],'class_gt_area':[96/(600*400)]}
S,meta=m.select_cached_universe([R],D)
check({a['image_id'] for a in S['annotations']}=={11},'Cached universe GT subset wrong')
check({i['id'] for i in S['images']}=={11},'Cached universe images wrong')
check(meta['uncached_raw_train_gt_count']==1,'Missing GT count not disclosed')
check(meta['uncached_raw_train_image_count']==1,'Missing image count not disclosed')
check(meta['raw_train_video_image_count']==2,'Wrong raw Train video census')
check(meta['certified_cached_image_count']==1,'Wrong cached frame census')
check(meta['coco_universe'].startswith('only certified'),'Scope description missing')
for alteration in [dict(R,video=5),dict(R,image_id=99),dict(R,class_gt=[0]),
                   dict(R,class_gt_area=[.3])]:
    try:m.select_cached_universe([alteration],D)
    except AssertionError:pass
    else:raise AssertionError('Invalid row not rejected '+str(alteration))
    N+=1
try:m.select_cached_universe([R,R],D)
except AssertionError:pass
else:raise AssertionError('Duplicate cached image allowed')
N+=1
BAD=dict(D)
BAD['images']=D['images']+[D['images'][0]]
try:m.select_cached_universe([R],BAD)
except AssertionError:pass
else:raise AssertionError('Duplicate official image ID allowed')
N+=1
# Source isolation: PATCH02 is an independent recovery entrypoint and never
# overwrites original RUN15A checkpoints/calibration/qualification.
source=p.read_text(encoding='utf-8');tree=ast.parse(source)
check('ORIG_OUT' in source and 'PINNED_PATCH01' in source,'Frozen origin/source lock missing')
check('run15a_patch02_trainonly_qualification.json' in source,'New output path missing')
check('score_threshold' not in source.lower() or 'm.CAL_THRESHOLDS' in source,'Threshold lock missing')
check('qualification already exists' in source.lower(),'Single-use gate missing')
check('no_new_model_training_or_calibration' in source,'No retraining provenance missing')
print(f'[PASS] Independent RUN15A PATCH02 recovery suite: {N} assertions; no SeaClips or checkpoint read')
