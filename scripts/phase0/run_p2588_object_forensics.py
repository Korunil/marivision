"""P-2588. TRAIN-allowlisted, CPU-only GT object forensics. V12 Stage 2 ZIP-native blind adjudication, two-tree rater/outcome separation, and audited cluster-merge protocol.

No detector or model imports; no model forward; no training; no TUNE/EVAL/CAL/QUAL.
Stage BUILD yields *provisional* GT links, never verified physical-object IDs.
Stage ADJUDICATE requires explicit human evidence to turn reviewed clusters into IDs.
No architecture GO. Offline scientific and source semantics remain conditional.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, random, re, shutil, statistics, uuid, zipfile
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = (2,3,4,5,7,8,13,14,16,17,18,19,20,24,25,31,35)
FORBIDDEN = frozenset((6,29,36,37,38,40,41,42,43,44,45,47,1,9,10,22,26,27,28,32,34,11,12,15,21,23,30,33,39,46))
TRAIN_COCO_SHA='edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35'
SPATIAL_CSV_SHA='941110163dede4fb1aa14c8b7e1a0e8d48cecde061ab33c6adafdaab78a32ad3'
SPATIAL_JSON_SHA='fb2b486cea80bc4619c9e9707bf1806ced1751bbc5a5516eebc6f8dcd382a5a5'
FN_JSON_SHA='358c8d277a3ef354477c0b7cea7afd3fffd6c4a6960281f5d8ad063aec67c5c3'
STAGE1_V11_SCHEMA='P2588_TRAIN_ONLY_OBJECT_FORENSIC_REVIEW_V11_ZIP_NATIVE'
STAGE1_V11_RUNNER_SHA256='9b624bb5d682835074670fe054ff32fc7d53ad1dcb197e6e348ef3e1402daf39'
SCHEMA='P2588_TRAIN_ONLY_OBJECT_FORENSIC_REVIEW_V12_ZIP_NATIVE'
ACCEPTED_STAGE1_BUILD_SCHEMAS=(STAGE1_V11_SCHEMA,SCHEMA)
MAX_QA_PANELS_DEFAULT=1025
SEED=20261014
# PREDECLARED, GT-only linking gates. These are provisional association heuristics.
LINK_MAX_DISPLACEMENT_GT_GEOMEAN=2.5
LINK_MIN_DISPLACEMENT_PX=4.0
LINK_MAX_LOG_AREA_RATIO=math.log(2.0)
LINK_AMBIGUITY_MARGIN=0.25
LINK_MAX_GAP_SAMPLED=1
MAX_BLIND_CASES=40
CLASS_NAMES=('ANIMAL','BOAT','MARINE_MARKER','OBJECT','LEISURE_VEHICLE','SAILING_VESSEL','SHIP')


def require(x,why):
    if not x: raise AssertionError('[P2588 HOLD] '+str(why))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for part in iter(lambda:s.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def current_runner_sha256():
    return sha(Path(__file__).resolve(strict=True))


def source_paths(project,coco=None):
    base=Path(project)
    spatial=base/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct13_spatial_matched_control'
    fn=base/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct13_recoverability_gate'
    return {'coco':Path(coco or r'D:\MARIVISION\seaclips-train.json'),
            'csv':spatial/'train_spatial_units.csv',
            'spatial':spatial/'train_spatial_diagnostic.json',
            'fn':fn/'train_fn_recoverability.json'}


PUBLIC_SOURCE_ROLES={
    'coco':'PINNED_TRAIN_COCO_JSON',
    'csv':'PINNED_TRAIN_UNIT_CENSUS_RECEIPT',
    'spatial':'PINNED_TRAIN_DIAGNOSTIC_RECEIPT',
    'fn':'PINNED_TRAIN_RECOVERABILITY_RECEIPT',
}


def verify_sources(paths):
    for name,expected in [('coco',TRAIN_COCO_SHA),('csv',SPATIAL_CSV_SHA),
                          ('spatial',SPATIAL_JSON_SHA),('fn',FN_JSON_SHA)]:
        p=paths[name]
        require(p.is_file(),f'Missing source {name}: {p}')
        got=sha(p)
        require(got==expected,f'{name} SHA mismatch: {got} != {expected}')
    return {k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}


def public_source_pins(pins):
    """Redact local filesystem paths from reviewer-facing provenance; retain logical roles and SHA-256 only."""
    return {k:{'source_role':PUBLIC_SOURCE_ROLES.get(k,str(k)),'sha256':str(v['sha256'])}
            for k,v in pins.items()}


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows,columns):
    with Path(path).open('w',newline='',encoding='utf-8') as s:
        w=csv.DictWriter(s,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def load_whitelist(csvpath,spatialpath,fnpath):
    rows=read_csv(csvpath)
    require(len(rows)==1025,'Expected exactly 1025 GT units')
    images=defaultdict(list);seen_ann=set();counts=Counter();videos=defaultdict(set)
    for x in rows:
        vid=int(x['video']); frame=int(x['frame']); iid=int(x['image_id']);aid=int(x['annotation_id'])
        cl=int(x['class']);tp=int(x['is_tp'])
        require(vid in TRAIN and vid not in FORBIDDEN,'Disallowed video in CSV')
        require(frame in range(0,160,10) and 0<=cl<7 and tp in (0,1),'CSV bad frame/class/TP')
        require(aid not in seen_ann,'Duplicate annotation_id');seen_ann.add(aid)
        require(x['class_name']==CLASS_NAMES[cl],'Class name/logit mismatch')
        images[iid].append(x);videos[vid].add((frame,iid));counts['tp' if tp else 'fn']+=1
    # The GT-unit CSV omits ALL zero-GT frames: 248 GT-bearing images, not 272.
    # 272-image universe must be reconstructed from pinned COCO TRAIN *image metadata* below.
    require(len(images)==248 and len(videos)==16,'GT-unit CSV image census drift (248 expected)')
    require(all(len(v)==16 for vid,v in videos.items() if vid!=8),
            'Expected 16 GT-bearing sampled images for non-video-8 represented videos')
    require(len(videos[8])==8 and 31 not in videos,'Expected zero-GT video31 and 8 zero-GT video8 frames')
    require(counts==Counter(tp=754,fn=271),'Historical TP/FN drift')
    js=json.loads(Path(spatialpath).read_text(encoding='utf-8'))
    require(js['TRAIN_only'] and js['images']==272 and js['GT_units']==1025,'Spatial report drift')
    require(js['baseline']=={'tp':754,'fp':1013,'fn':271},'Native baseline drift')
    fn=json.loads(Path(fnpath).read_text(encoding='utf-8'))
    require(fn['TRAIN_only'] and len(fn['fn_events'])==271,'FN receipt drift')
    actual={(int(x['video']),int(x['frame']),int(x['image_id']),int(x['gt_index']))
            for x in rows if int(x['is_tp'])==0}
    expected={(int(x['video']),int(x['frame']),int(x['image_id']),int(x['g'])) for x in fn['fn_events']}
    require(actual==expected,'Exact historical FN identity mismatch')
    return rows,images,videos


def source_image_join_diagnostic(raw,unit_rows):
    """GT-only source-identity inventory; cache frame indices are NOT raw COCO frame_id.

    Does not infer zero-GT frame image IDs or retrieve images outside CSV allowlist.
    """
    by_id={}; duplicate_source=[]
    for im in raw['images']:
        iid=int(im['id'])
        if iid in by_id:duplicate_source.append(iid)
        by_id[iid]=im
    row_slots={};conflicts=[]
    for r in unit_rows:
        iid=int(r['image_id']);slot=(int(r['video']),int(r['frame']))
        if iid in row_slots and row_slots[iid]!=slot:conflicts.append(iid)
        row_slots[iid]=slot
    samples=defaultdict(list);wrong_video=[];absent=[];missing_frame_id=[]
    for iid,(vid,cached_frame) in sorted(row_slots.items()):
        meta=by_id.get(iid)
        if meta is None:absent.append(iid);continue
        if int(meta.get('video_id',-1))!=vid:wrong_video.append(iid)
        if 'frame_id' not in meta:missing_frame_id.append(iid);continue
        raw_frame=int(meta['frame_id'])
        samples[vid].append((cached_frame,raw_frame,iid))
    per_video={}
    for vid in TRAIN:
        ss=sorted(samples.get(vid,[]))
        offsets=sorted(set(rawf-cached for cached,rawf,iid in ss))
        per_video[str(vid)]={'gt_bearing_image_ids_verified':len(ss),
          'source_frame_offsets':offsets,'sample_pairs':[
           {'cached_frame':c,'coco_frame_id':f,'image_id':iid} for c,f,iid in ss[:2]],
          'source_frame_mapping_status':'CONSTANT_OFFSET' if len(offsets)==1 else
            ('NO_GT_ANCHORS' if not ss else 'NONCONSTANT_OFFSET_HOLD')}
    all_ann={};dup_ann=[]
    for a in raw['annotations']:
        aid=int(a['id'])
        if aid in all_ann:dup_ann.append(aid)
        all_ann[aid]=a
    expected_aids={int(r['annotation_id']) for r in unit_rows}
    absent_aids=expected_aids-set(all_ann)
    annotation_wrong_image_or_class=[int(r['annotation_id']) for r in unit_rows
            if int(r['annotation_id']) in all_ann and
               (int(all_ann[int(r['annotation_id'])]['image_id'])!=int(r['image_id']) or
                int(all_ann[int(r['annotation_id'])]['category_id'])!=int(r['class'])+1)]
    extra_ann=[int(a['id']) for a in raw['annotations'] if int(a['image_id']) in row_slots
               and int(a['id']) not in expected_aids]
    out={'status':'DIAGNOSTIC_ONLY_NO_SOURCE_CHANGE',
      'source_join_rule':'EXACT_PINNED_CACHED_IMAGE_ID_AND_ANNOTATION_ID',
      'cached_frame_not_assumed_equal_to_coco_frame_id':True,
      'frozen_cached_sampled_frame_count':272,
      'gt_bearing_csv_image_count':len(row_slots),
      'coco_images_with_exact_cached_ids':len(row_slots)-len(absent),
      'missing_exact_cached_image_ids':absent[:30],
      'missing_exact_cached_image_count':len(absent),
      'wrong_video_image_ids':wrong_video[:30],
      'missing_source_frame_id_count':len(missing_frame_id),
      'duplicate_source_image_ids_count':len(duplicate_source),
      'conflicting_csv_image_slots_count':len(conflicts),
      'exact_annotation_ids_in_full_source':len(expected_aids)-len(absent_aids),
      'missing_exact_annotation_ids_count':len(absent_aids),
      'duplicate_source_annotation_ids_count':len(dup_ann),
      'wrong_image_or_category_annotation_ids_count':len(annotation_wrong_image_or_class),
      'extra_annotations_on_exact_gt_bearing_images_count':len(extra_ann),
      'per_video':per_video,
      'zero_gt_cached_frames_NOT_resolved_to_coco_image_records':24,
      'pinned_source_categories_raw':raw['categories'],
      'source_file_name_samples':[{'image_id':iid,'file_name':str(by_id.get(iid,{}).get('file_name',''))} for iid in sorted(row_slots)[:3]],
      'warning':'Image IDs and annotation IDs are source-authenticated. Cached sampling index is NOT presumed to equal COCO frame_id.'}
    return out


def checked_authorized_coco(cocopath,unit_rows,allowed_images,*,raw=None):
    """Join by the exact source image/annotation IDs previously validated by native R50.

    The OCT13 native spatial audit directly looked up raw['images'] by the
    canonical cache image_id; it did NOT assert raw frame_id equals cache index.
    Never search global COCO using video_id + cache frame=0,10,...,150.
    """
    if raw is None:raw=json.loads(Path(cocopath).read_text(encoding='utf-8'))
    categories={int(x['id']):str(x['name']).upper() for x in raw['categories']}
    require(categories=={i+1:x for i,x in enumerate(CLASS_NAMES)},'Seven-class COCO label mapping mismatch')
    inv=source_image_join_diagnostic(raw,unit_rows)
    require(inv['gt_bearing_csv_image_count']==len(allowed_images),
            'Source image allowlist count inconsistent with frozen CSV')
    for field in ('missing_exact_cached_image_count','wrong_image_or_category_annotation_ids_count',
        'missing_source_frame_id_count','duplicate_source_image_ids_count',
        'duplicate_source_annotation_ids_count','conflicting_csv_image_slots_count',
        'missing_exact_annotation_ids_count','extra_annotations_on_exact_gt_bearing_images_count'):
        require(inv[field]==0,'Exact source ID join drift: '+field+'='+str(inv[field]))
    require(not inv['wrong_video_image_ids'],'Source COCO video_id differs from frozen cache')
    for vid,stat in inv['per_video'].items():
        if stat['gt_bearing_image_ids_verified']:
            require(stat['source_frame_mapping_status']=='CONSTANT_OFFSET',
                    'COCO frame_id minus cached index not constant for video '+vid)
    # No invention of image records or identities for the 24 zero-GT cache frames.
    ids=set(int(i) for i in allowed_images)
    imgs={int(im['id']):dict(im) for im in raw['images'] if int(im['id']) in ids}
    by_image_slot={}
    for r in unit_rows:
        iid=int(r['image_id']);slot=(int(r['video']),int(r['frame']))
        require(iid in imgs,'GT-bearing COCO image ID absent')
        if iid in by_image_slot:require(by_image_slot[iid]==slot,'Conflicting frozen image index')
        by_image_slot[iid]=slot
    for iid,(vid,frame) in by_image_slot.items():
        imgs[iid]['_p2588_cached_frame_idx']=frame
        require(int(imgs[iid]['video_id'])==vid,'Source video mismatch')
        require(int(imgs[iid]['frame_id'])-frame==inv['per_video'][str(vid)]['source_frame_offsets'][0],
                'Source frame offset mismatch')
    aids={int(r['annotation_id']) for r in unit_rows}
    anns={int(a['id']):a for a in raw['annotations'] if int(a['id']) in aids}
    require(set(anns)==aids,'COCO missing exact 1025 GT annotation IDs')
    for r in unit_rows:
        a=anns[int(r['annotation_id'])]
        require(int(a['image_id'])==int(r['image_id']) and int(a['category_id'])==int(r['class'])+1,
                'Exact source annotation image/class alignment drift')
    return imgs,anns


def inventory_allowlist(cocopath,unit_rows):
    """Read-only source-ID audit. Do not assume COCO frame_id equals cached j."""
    raw=json.loads(Path(cocopath).read_text(encoding='utf-8'))
    return source_image_join_diagnostic(raw,unit_rows)


def materialize_gt(units,imgs,anns):
    byimg=defaultdict(list);out=[]
    for row in units:
        aid=int(row['annotation_id']);iid=int(row['image_id']);vid=int(row['video']);frame=int(row['frame'])
        image=imgs[iid];ann=anns[aid]
        w=float(image['width']);h=float(image['height'])
        x,y,bw,bh=map(float,ann['bbox'])
        require(w>0 and h>0 and bw>0 and bh>0,'Invalid original COCO bbox')
        require(x>=-0.01 and y>=-0.01 and x+bw<=w+0.01 and y+bh<=h+0.01,'Out-of-image GT bbox')
        require(int(image['video_id'])==vid and int(image.get('_p2588_cached_frame_idx',frame))==frame,
                'Frozen cached frame slot/source metadata mismatch')
        require(int(ann['image_id'])==iid and int(ann['category_id'])==int(row['class'])+1,'GT category/image mismatch')
        reported=float(row['gt_area_source_px2'])
        require(abs(bw*bh-reported)<=max(.05,.002*reported),'Frozen GT pixel area mismatch')
        z={'annotation_id':aid,'image_id':iid,'video':vid,'frame':frame,'class':int(row['class']),
           'is_tp':int(row['is_tp']),'gt_index':int(row['gt_index']),
           'x':x,'y':y,'w':bw,'h':bh,'area':bw*bh,'cx':x+bw/2,'cy':y+bh/2,
           'source_w':w,'source_h':h,'camera':str(image.get('camera_sensor','UNKNOWN')),
           'sea_state':str(image.get('sea_state','UNKNOWN')),
           'image_filename':str(image.get('file_name','')),
           'best_raw_query':str(row['best_raw_query']),
           'signed_center_x_over_gt_w':float(row['signed_center_x_over_gt_w']),
           'signed_center_y_over_gt_h':float(row['signed_center_y_over_gt_h']),
           'log_pred_w_over_gt_w':float(row['log_pred_w_over_gt_w']),
           'log_pred_h_over_gt_h':float(row['log_pred_h_over_gt_h'])}
        out.append(z);byimg[iid].append(z)
    require(len(out)==1025,'GT census mismatch')
    return out,byimg


def coco_track_key(ann):
    """Never assume an annotation's numeric id is a persistent track id."""
    names=('track_id','instance_id','object_track_id','persistent_object_id')
    return tuple((k,str(ann[k])) for k in names if k in ann and str(ann[k]).strip() not in ('','None','-1'))


def gt_neighbors(units,byimg):
    rows=[]
    for z in units:
        other=[q for q in byimg[z['image_id']] if q['annotation_id']!=z['annotation_id']]
        if other:
            d=min(math.hypot(z['cx']-p['cx'],z['cy']-p['cy']) for p in other)
        else:d=None
        scale=math.sqrt(z['area'])
        rows.append({'annotation_id':z['annotation_id'],'video':z['video'],'frame':z['frame'],
                     'image_id':z['image_id'],'class':z['class'],
                     'gt_w_source_px':z['w'],'gt_h_source_px':z['h'],'gt_area_source_px2':z['area'],
                     'cx_source_px':z['cx'],'cy_source_px':z['cy'],
                     'nearest_anyclass_gt_center_source_px':d,
                     'nearest_anyclass_gt_center_in_gt_geomean':(d/scale if d is not None else None),
                     'n_gt_in_frame':len(byimg[z['image_id']])})
    return rows


def iou_xywh(a,b):
    x1=max(a['x'],b['x']);y1=max(a['y'],b['y'])
    x2=min(a['x']+a['w'],b['x']+b['w']);y2=min(a['y']+a['h'],b['y']+b['h'])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1e-12,a['area']+b['area']-inter)


def proposal_cost(a,b):
    if a['video']!=b['video'] or a['class']!=b['class'] or b['frame']-a['frame']!=10:
        return None
    require((a['source_w'],a['source_h'])==(b['source_w'],b['source_h']),
            'Video dimensions changed; cannot use pixel linking')
    dist=math.hypot(a['cx']-b['cx'],a['cy']-b['cy'])
    gm=math.sqrt(math.sqrt(a['area']*b['area']))
    maxdist=max(LINK_MIN_DISPLACEMENT_PX,LINK_MAX_DISPLACEMENT_GT_GEOMEAN*gm)
    loga=abs(math.log(a['area']/b['area']))
    if dist>maxdist or loga>LINK_MAX_LOG_AREA_RATIO:return None
    overlap=iou_xywh(a,b)
    score=dist/max(gm,1e-9)+.35*loga+.1*(1-overlap)
    return {'source_annotation_id':a['annotation_id'],'target_annotation_id':b['annotation_id'],
            'video':a['video'],'class':a['class'],'from_frame':a['frame'],'to_frame':b['frame'],
            'displacement_px':dist,'displacement_over_gt_size':dist/max(gm,1e-9),
            'log_area_ratio_abs':loga,'gt_box_iou':overlap,'cost':score}


def provisional_edges(units):
    by=defaultdict(list)
    for z in units:by[(z['video'],z['class'],z['frame'])].append(z)
    edges=[];accepted=[];ambiguous=0
    for (vid,c,frame),src in sorted(by.items()):
        targets=by.get((vid,c,frame+10),[])
        candidates=[v for a in src for b in targets if (v:=proposal_cost(a,b)) is not None]
        row_src=defaultdict(list);row_dst=defaultdict(list)
        for e in candidates:
            row_src[e['source_annotation_id']].append(e);row_dst[e['target_annotation_id']].append(e)
        for e in candidates:
            x=sorted(row_src[e['source_annotation_id']],key=lambda t:(t['cost'],t['target_annotation_id']))
            y=sorted(row_dst[e['target_annotation_id']],key=lambda t:(t['cost'],t['source_annotation_id']))
            mx=(x[1]['cost']-x[0]['cost']) if len(x)>1 else float('inf')
            my=(y[1]['cost']-y[0]['cost']) if len(y)>1 else float('inf')
            mutual=x[0] is e and y[0] is e
            strong=mutual and mx>=LINK_AMBIGUITY_MARGIN and my>=LINK_AMBIGUITY_MARGIN
            e={**e,'second_cost_margin_from':(mx if math.isfinite(mx) else None),
                'second_cost_margin_to':(my if math.isfinite(my) else None),
                'linkage_status':'PROPOSED_UNVERIFIED' if strong else 'AMBIGUOUS_NOT_JOINED'}
            edges.append(e)
            if strong:accepted.append(e)
            else:ambiguous+=1
    seen_from=[x['source_annotation_id'] for x in accepted]
    seen_to=[x['target_annotation_id'] for x in accepted]
    require(len(set(seen_from))==len(seen_from) and len(set(seen_to))==len(seen_to),
            'Proposed edges not one-to-one')
    return edges,accepted,ambiguous


def provisional_components(units,edges):
    parent={z['annotation_id']:z['annotation_id'] for z in units}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for e in edges:
        a=root(e['source_annotation_id']);b=root(e['target_annotation_id']);parent[b]=a
    grouped=defaultdict(list)
    for z in units: grouped[root(z['annotation_id'])].append(z)
    clusters=[];lookup={}
    for i,gg in enumerate(sorted(grouped.values(),key=lambda q:(min(z['video'] for z in q),
                             min(z['class'] for z in q),min(z['annotation_id'] for z in q)))):
        cid=f'P{i+1:04d}'; v={z['video'] for z in gg}; c={z['class'] for z in gg}
        require(len(v)==1 and len(c)==1,'Cross-video/class provisional cluster')
        require(len({z['frame'] for z in gg})==len(gg),'Duplicate frame in provisional cluster')
        for z in gg:lookup[z['annotation_id']]=cid
        clusters.append({'provisional_cluster_id':cid,'video':next(iter(v)),'class':next(iter(c)),
                         'class_name':CLASS_NAMES[next(iter(c))],'n_observations':len(gg),
                         'first_frame':min(z['frame'] for z in gg),'last_frame':max(z['frame'] for z in gg),
                         'annotation_ids':';'.join(str(z['annotation_id']) for z in sorted(gg,key=lambda z:z['frame'])),
                         'verification_status':'PROVISIONAL_UNVERIFIED',
                         'physical_object_id':'','identity_evidence':'', 'verified_by':''})
    return clusters,lookup


def image_path(image_root,filename):
    root=Path(image_root).resolve(strict=True)
    # Windows absolute paths inside COCO metadata are not accepted as relative image paths.
    q=Path(str(filename).replace('\\','/'))
    require(not q.is_absolute() and not re.match(r'^[A-Za-z]:',str(q)),'COCO image path absolute')
    p=(root/q).resolve()
    require(p.is_relative_to(root),'COCO image path escapes authorized root')
    require(p.is_file(),'Allowed TRAIN image missing: '+str(p))
    return p


def verify_train_images(image_root, imgs, expected_ids):
    """All 248 authorized GT-bearing source images: containment, header dimensions and provenance.

    COCO is SHA-pinned; source pixels are NOT hash-pinned independently. Dimensions
    and validated confined paths are a necessary check, not proof of pixel identity.
    """
    require(image_root is not None and str(image_root).strip(),
            '--images-root required for P2588 preflight and build')
    from PIL import Image
    root=Path(image_root).resolve(strict=True)
    require(root.is_dir(),'Image root must be an existing directory')
    expected=set(map(int,expected_ids))
    require(set(imgs)==expected and len(expected)==248,
            'Authenticated image header allowlist must contain 248 exact GT-bearing image IDs')
    paths={}; seen_paths=set(); sampled=[]
    for iid in sorted(expected):
        meta=imgs[iid]
        filename=str(meta.get('file_name',''))
        require(bool(filename),'Missing file_name for source image '+str(iid))
        path=image_path(root,filename)
        require(path not in seen_paths,'Two distinct source image IDs point to the same file: '+str(path))
        seen_paths.add(path)
        with Image.open(path) as im:
            actual=tuple(map(int,im.size))
            required=(int(meta['width']),int(meta['height']))
            require(actual==required,
                    'TRAIN source-image dimensions differ from pinned COCO image_id='+str(iid)+
                    ' expected='+str(required)+' actual='+str(actual))
            require(im.format in ('JPEG','PNG','BMP','TIFF','WEBP'),
                    'Unrecognized source-image format image_id='+str(iid))
        paths[iid]=path
        if len(sampled)<3:sampled.append({'image_id':iid,'file_name':filename,'dimensions':list(actual)})
    return {'n_verified_GT_bearing_source_image_headers':len(paths),
            'image_root':str(root),'validation':'EXACT_COCO_FILE_NAME_PLUS_DIMENSION_CHECK; NO_INDEPENDENT_PIXEL_HASH',
            'sample_images':sampled}


class TrainImagesZip:
    """Confined read-only archive resolver for EXACT GT-bearing COCO image IDs.

    Indexing central-directory *names* is allowed; no other split's image bytes
    are opened, decoded, extracted, or loaded. Never guess via basename alone.
    """
    def __init__(self, path):
        self.path=Path(path).resolve(strict=True)
        require(self.path.is_file() and zipfile.is_zipfile(self.path),
                'SeaClips ZIP missing/invalid: '+str(self.path))
        self.archive=None
        self.members={}
        self.image_members={}
        self.sha_by_id={}
        self.initial_stat=(self.path.stat().st_size,self.path.stat().st_mtime_ns)

    def __enter__(self):
        self.archive=zipfile.ZipFile(self.path,'r',allowZip64=True)
        self.members={}
        for info in self.archive.infolist():
            if info.is_dir():continue
            name=info.filename.replace('\\','/')
            if name.startswith('./'):name=name[2:]
            # Never use an unsafe member path even though no extraction is performed.
            require(bool(name) and not name.startswith('/') and
                    not re.match(r'^[A-Za-z]:',name) and
                    all(p not in ('','.','..') for p in name.split('/')),
                    'Unsafe ZIP member name encountered')
            require(name not in self.members,'Duplicate ZIP member path: '+name)
            self.members[name]=info
        return self

    def __exit__(self,*exc):
        if self.archive is not None:self.archive.close()
        self.archive=None

    @staticmethod
    def clean_coco(filename):
        filename=str(filename).replace('\\','/')
        require(bool(filename.strip()) and not filename.startswith('/') and
                not re.match(r'^[A-Za-z]:',filename) and
                all(p not in ('','.','..') for p in filename.split('/')),
                'Unsafe source COCO file_name')
        return filename

    def resolve(self,imgs,expected_ids):
        expected=set(map(int,expected_ids))
        require(len(expected)==248 and set(map(int,imgs))==expected,
                'ZIP resolution restricted to 248 exact GT-bearing image IDs')
        mapping={};selected=set()
        for iid in sorted(expected):
            fn=self.clean_coco(imgs[iid].get('file_name',''))
            options=[name for name in self.members if name==fn or name.endswith('/'+fn)]
            # Dataset has TRAIN/VALIDATION/TEST in a single archive. Match must
            # select an explicitly TRAIN-scoped member; no cross-split byte reads.
            options=[name for name in options if 'train' in name.lower().split('/')[:-1]]
            require(len(options)==1,
                    'TRAIN ZIP source member must resolve uniquely for image_id='+str(iid)+
                    ' file_name='+fn+' matches='+str(options[:5]))
            member=options[0]
            require(member not in selected,'Different image IDs alias one ZIP member: '+member)
            selected.add(member)
            info=self.members[member]
            require(0 < info.file_size <= 75*1024*1024,
                    'TRAIN ZIP member size outside bounded image limit: '+member)
            mapping[iid]=member
        self.image_members=mapping
        return mapping

    def read_bytes(self,iid):
        require(self.archive is not None,'ZIP image source not opened')
        require(iid in self.image_members,'Unallowlisted image_id ZIP read denied: '+str(iid))
        require((self.path.stat().st_size,self.path.stat().st_mtime_ns)==self.initial_stat,
                'SeaClips ZIP changed after the ZIP reader was opened')
        info=self.members[self.image_members[iid]]
        b=self.archive.read(info)  # Built-in CRC check; no extraction, no other splits.
        require(len(b)==info.file_size,'ZIP decompression length mismatch')
        h=hashlib.sha256(b).hexdigest()
        if iid in self.sha_by_id:
            require(h==self.sha_by_id[iid],'TRAIN ZIP source image content changed during build')
        return b,h

    def rgb(self,meta):
        from PIL import Image
        iid=int(meta['id'])
        b,h=self.read_bytes(iid)
        with Image.open(io.BytesIO(b)) as image:
            require(image.size==(int(meta['width']),int(meta['height'])),
                    'TRAIN ZIP image dimensions changed for image_id='+str(iid))
            return image.convert('RGB')


def verify_train_images_zip(source,imgs,expected_ids):
    """Validate EVERY authorized GT image through original ZIP bytes and PIL.

    SHA-256 records enable within-run identity consistency; they do not certify
    an independent publisher-signed pixel hash for the full archive.
    """
    from PIL import Image
    mapping=source.resolve(imgs,expected_ids)
    digest=hashlib.sha256(); samples=[]
    for iid in sorted(mapping):
        b,h=source.read_bytes(iid)
        with Image.open(io.BytesIO(b)) as im:
            dims=tuple(map(int,im.size))
            expected=(int(imgs[iid]['width']),int(imgs[iid]['height']))
            require(dims==expected,
                    'TRAIN ZIP image dimensions differ from pinned COCO image_id='+str(iid))
            require(im.format in ('JPEG','PNG','BMP','TIFF','WEBP'),
                    'Unrecognized TRAIN ZIP source image format for '+str(iid))
            im.load()  # Decoder integrity check, no disk extraction.
        source.sha_by_id[iid]=h
        digest.update(f'{iid}|{mapping[iid]}|{h}\n'.encode('utf-8'))
        if len(samples)<3:
            samples.append({'image_id':iid,'coco_file_name':imgs[iid]['file_name'],
                            'zip_member':mapping[iid],'dimensions':list(dims)})
    return {'n_verified_GT_bearing_source_image_headers':len(mapping),
            'image_zip':str(source.path),
            'verification':'EXACT_CACHED_ID_COCO_FILE_NAME_UNIQUE_TRAIN_MEMBER_DIMENSIONS_AND_ZIP_CRC',
            'authorized_image_content_receipt_sha256':digest.hexdigest(),
            'pixel_sha_is_intra_run_not_independent_source_authentication':True,
            'sample_images':samples}


def load_source_rgb(image_source,meta):
    from PIL import Image
    if isinstance(image_source,TrainImagesZip):
        return image_source.rgb(meta)
    path=image_path(image_source,meta['file_name'])  # Unit-test legacy only.
    with Image.open(path) as raw:return raw.convert('RGB')


def private_output_destination(args):
    require(args.sealed_root is not None and str(args.sealed_root).strip(),
            '--sealed-root required; outcome files must not be in identity-review package')
    share_root=Path(args.output).resolve()
    private_root=Path(args.sealed_root).resolve()
    require(private_root!=share_root and not private_root.is_relative_to(share_root)
            and not share_root.is_relative_to(private_root),
            'Private sealed root and reviewer-facing output must be in separate non-nested trees')
    # Reject sharing the same parent with separate names only if paths overlap; user must
    # distribute ONLY P2588_identity_build, never its parent nor private outcomes root.
    return private_root/'P2588_sealed_cluster_outcomes'


def private_adjudication_destination(args):
    require(args.sealed_root is not None and str(args.sealed_root).strip(),
            '--sealed-root required; Stage 2 outcome and case-key files must not be in rater-facing blind package')
    share_root=Path(args.output).resolve()
    private_root=Path(args.sealed_root).resolve()
    require(private_root!=share_root and not private_root.is_relative_to(share_root)
            and not share_root.is_relative_to(private_root),
            'Private sealed root and rater-facing output must be in separate non-nested trees')
    return private_root/'P2588_sealed_adjudication_outcomes'


def ensure_output_unused(dest):
    require(not dest.exists() and not Path(str(dest)+'.staging').exists(),'Immutable output already exists or stale staging')


ACTIVE_INVOCATION_TOKEN=None
ACTIVE_CREATED_STAGING=None
ACTIVE_PUBLISHED_FINAL=None
ACTIVE_FAILED_CLEANUP=None
ACTIVE_VERIFIED_ABSENT_STAGING=None


def write_invocation_manifest(folder,files,invocation_token=None,runner_sha256=None):
    tok=invocation_token if invocation_token is not None else ACTIVE_INVOCATION_TOKEN
    rsha=runner_sha256 if runner_sha256 is not None else current_runner_sha256()
    payload={'schema':SCHEMA,'runner_sha256':str(rsha),'files':files}
    if tok is not None:
        payload['invocation_token']=str(tok)
    (Path(folder)/'SHA256_MANIFEST.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


def is_owned_by_invocation(folder,invocation_token,owned_paths=None,verified_absent_staging=None):
    path=Path(folder)
    if owned_paths is not None and any(path==Path(x) for x in owned_paths):
        return True
    if verified_absent_staging is not None and any(path==Path(x) for x in verified_absent_staging):
        try:
            if path.is_dir() and not any(path.iterdir()):
                return True
        except Exception:
            pass
    if invocation_token is None:return False
    mpath=path/'SHA256_MANIFEST.json'
    if not mpath.is_file():return False
    try:
        data=json.loads(mpath.read_text(encoding='utf-8'))
        return str(data.get('invocation_token',''))==str(invocation_token)
    except Exception:
        return False


def record_failed_cleanup(path):
    try:
        if ACTIVE_FAILED_CLEANUP is not None and path not in ACTIVE_FAILED_CLEANUP:
            ACTIVE_FAILED_CLEANUP.append(path)
    except BaseException:
        pass


def remove_verified_directory(path):
    p=Path(path)
    if not p.exists():return
    try:
        shutil.rmtree(p)
    except BaseException as cleanup_exc:
        record_failed_cleanup(p)
        raise RuntimeError(f'[P2588 ROLLBACK FAILURE] Failed to remove owned outputs: {p}: {cleanup_exc}') from cleanup_exc
    if p.exists():
        record_failed_cleanup(p)
        raise RuntimeError(f'[P2588 ROLLBACK FAILURE] Failed to remove owned outputs: {p}: still exists after rmtree')


def tmp_output(dest):
    ensure_output_unused(dest)
    tmp=Path(str(dest)+'.staging')
    registered=False
    if ACTIVE_VERIFIED_ABSENT_STAGING is not None and tmp not in ACTIVE_VERIFIED_ABSENT_STAGING:
        ACTIVE_VERIFIED_ABSENT_STAGING.append(tmp)
    # Zero intervening statements between mkdir(exist_ok=False) and try:;
    # ACTIVE_VERIFIED_ABSENT_STAGING also records that tmp was verified absent prior to mkdir.
    tmp.mkdir(parents=True,exist_ok=False)
    try:
        if ACTIVE_CREATED_STAGING is not None and tmp not in ACTIVE_CREATED_STAGING:
            ACTIVE_CREATED_STAGING.append(tmp)
            registered=True
        tok=ACTIVE_INVOCATION_TOKEN if ACTIVE_INVOCATION_TOKEN is not None else uuid.uuid4().hex
        write_invocation_manifest(tmp,{},invocation_token=tok)
        require(is_owned_by_invocation(tmp,tok),
                'Staging ownership manifest verification failed for: '+str(tmp))
    except BaseException as init_exc:
        try:
            remove_verified_directory(tmp)
        except BaseException as rb_exc:
            raise RuntimeError(f'{rb_exc} (while cleaning up staging directory after: {init_exc})') from init_exc
        if registered and ACTIVE_CREATED_STAGING is not None:
            try:
                if tmp in ACTIVE_CREATED_STAGING:
                    ACTIVE_CREATED_STAGING.remove(tmp)
            except BaseException:
                pass
        raise
    return tmp


def prepare_output_manifest(tmp,invocation_token=None):
    if invocation_token is None and (Path(tmp)/'SHA256_MANIFEST.json').is_file():
        try:
            invocation_token=json.loads((Path(tmp)/'SHA256_MANIFEST.json').read_text(encoding='utf-8')).get('invocation_token')
        except Exception:
            invocation_token=None
    manifest={str(p.relative_to(tmp)).replace('\\','/') :sha(p) for p in sorted(tmp.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    write_invocation_manifest(tmp,manifest,invocation_token=invocation_token)
    if invocation_token is not None:
        require(is_owned_by_invocation(tmp,invocation_token),
                'Finalized ownership manifest verification failed for: '+str(tmp))
    return manifest


def publish_staged_output(tmp,dest,published_tracker=None):
    src_tmp=Path(tmp);dst_final=Path(dest)
    require(not dst_final.exists(),'Refusing to overwrite existing destination during publish: '+str(dst_final))
    src_tmp.rename(dst_final)
    registered_tracker=False
    registered_active=False
    try:
        if published_tracker is not None and dst_final not in published_tracker:
            published_tracker.append(dst_final)
            registered_tracker=True
        if ACTIVE_PUBLISHED_FINAL is not None and dst_final not in ACTIVE_PUBLISHED_FINAL:
            ACTIVE_PUBLISHED_FINAL.append(dst_final)
            registered_active=True
    except BaseException as pub_exc:
        try:
            remove_verified_directory(dst_final)
        except BaseException as rb_exc:
            raise RuntimeError(f'{rb_exc} (while cleaning up published directory after: {pub_exc})') from pub_exc
        if registered_tracker and published_tracker is not None:
            try:
                if dst_final in published_tracker:published_tracker.remove(dst_final)
            except BaseException:
                pass
        if registered_active and ACTIVE_PUBLISHED_FINAL is not None:
            try:
                if dst_final in ACTIVE_PUBLISHED_FINAL:ACTIVE_PUBLISHED_FINAL.remove(dst_final)
            except BaseException:
                pass
        raise


def rollback_owned_outputs(candidates,invocation_token,owned_paths=None,already_failed=None,verified_absent_staging=None):
    errors=[]
    seen=set()
    failed_set={Path(x) for x in (already_failed or ())}
    for p in candidates:
        path=Path(p)
        if path in seen or path in failed_set or not path.exists():continue
        seen.add(path)
        if not is_owned_by_invocation(path,invocation_token,owned_paths=owned_paths,verified_absent_staging=verified_absent_staging):
            continue
        try:
            shutil.rmtree(path)
        except BaseException as cleanup_exc:
            errors.append(f'{path}: {cleanup_exc}')
            continue
        if path.exists():
            errors.append(f'{path}: still exists after rmtree')
    if errors:
        raise RuntimeError('[P2588 ROLLBACK FAILURE] Failed to remove owned outputs: '+'; '.join(errors))


def verify_published_build_pair(dest,private_dest,expected_token=None,expected_runner_sha=None,allow_pinned_stage1_v11=False):
    rev_dir=Path(dest);priv_dir=Path(private_dest)
    require(rev_dir.is_dir(),f'Reviewer output directory missing: {rev_dir}')
    require(priv_dir.is_dir(),f'Private outcomes directory missing: {priv_dir}')
    require(not Path(str(rev_dir)+'.staging').exists(),f'Stale reviewer staging directory exists: {rev_dir}.staging')
    require(not Path(str(priv_dir)+'.staging').exists(),f'Stale private staging directory exists: {priv_dir}.staging')
    rev_mpath=rev_dir/'SHA256_MANIFEST.json'
    priv_mpath=priv_dir/'SHA256_MANIFEST.json'
    require(rev_mpath.is_file() and priv_mpath.is_file(),'Both published trees must contain SHA256_MANIFEST.json')
    rev_man=json.loads(rev_mpath.read_text(encoding='utf-8'))
    priv_man=json.loads(priv_mpath.read_text(encoding='utf-8'))
    allowed_schemas=ACCEPTED_STAGE1_BUILD_SCHEMAS if allow_pinned_stage1_v11 else (SCHEMA,)
    pub_schema=rev_man.get('schema')
    require(pub_schema in allowed_schemas and priv_man.get('schema')==pub_schema,'Published manifest schema mismatch')
    tok_rev=str(rev_man.get('invocation_token','')).strip()
    tok_priv=str(priv_man.get('invocation_token','')).strip()
    require(bool(tok_rev) and tok_rev==tok_priv,'Published two-tree invocation_token mismatch or missing')
    if expected_token is not None:
        require(tok_rev==str(expected_token),'Published invocation_token does not match active invocation')
    rsha_rev=str(rev_man.get('runner_sha256','')).strip()
    rsha_priv=str(priv_man.get('runner_sha256','')).strip()
    require(len(rsha_rev)==64 and rsha_rev==rsha_priv,'Published two-tree runner_sha256 mismatch or missing')
    if expected_runner_sha is not None:
        allowed_rshas={str(expected_runner_sha)}
        if allow_pinned_stage1_v11 and pub_schema==STAGE1_V11_SCHEMA:
            allowed_rshas.add(STAGE1_V11_RUNNER_SHA256)
        require(rsha_rev in allowed_rshas,'Published runner_sha256 does not match expected runner SHA-256')
    for tree,man in ((rev_dir,rev_man),(priv_dir,priv_man)):
        expected_files=man.get('files',{})
        require(isinstance(expected_files,dict) and len(expected_files)>0,f'Empty file manifest in {tree}')
        actual_files={str(p.relative_to(tree)).replace('\\','/'):sha(p)
                      for p in sorted(tree.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
        require(set(actual_files.keys())==set(expected_files.keys()),
                f'On-disk file set differs from SHA256_MANIFEST.json in {tree}')
        for rel,digest in expected_files.items():
            require(actual_files[rel]==digest,f'File SHA-256 mismatch in {tree}: {rel}')
    rep=json.loads((rev_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
    sealed_readme=json.loads((priv_dir/'SEALED_README.json').read_text(encoding='utf-8'))
    require(rep.get('schema')==pub_schema and sealed_readme.get('schema')==pub_schema,'Report/README schema mismatch')
    require(rep.get('runner_sha256')==rsha_rev and sealed_readme.get('runner_sha256')==rsha_rev,
            'Report/README runner_sha256 mismatch with manifest')
    n_clusters=int(rep['n_provisional_clusters_NOT_independent_objects'])
    require(n_clusters>0 and int(sealed_readme['n_provisional_clusters'])==n_clusters,
            'Reviewer report and SEALED_README cluster counts disagree')
    qa_rows=read_csv(rev_dir/'identity_qa_manifest.csv')
    tpl_rows=read_csv(rev_dir/'identity_review_template.csv')
    out_rows=read_csv(priv_dir/'SEALED_cluster_outcomes_DO_NOT_SHARE_WITH_IDENTITY_REVIEWERS.csv')
    cids_qa=[r['provisional_cluster_id'] for r in qa_rows]
    cids_tpl=[r['provisional_cluster_id'] for r in tpl_rows]
    cids_out=[r['provisional_cluster_id'] for r in out_rows]
    require(len(cids_qa)==n_clusters and cids_qa==cids_tpl and cids_qa==cids_out,
            'Cluster ID sequence mismatch across reviewer manifests and sealed outcomes')
    panel_names=sorted(p.name for p in (rev_dir/'identity_qa_panels').glob('*.png'))
    require(panel_names==sorted(f'{cid}.png' for cid in cids_qa),
            'Rendered QA PNG panels do not match provisional cluster IDs 1-to-1')
    forbidden_tokens=('train_spatial_units.csv','train_spatial_diagnostic.json','train_fn_recoverability.json',
                      'SeaClips.zip','is_tp','n_tp','n_fn','best_raw_query',
                      'signed_center_x_over_gt_w','signed_center_y_over_gt_h',
                      'log_pred_w_over_gt_w','log_pred_h_over_gt_h')
    for fpath in sorted(rev_dir.rglob('*')):
        if fpath.is_file() and fpath.suffix in ('.csv','.json','.md'):
            txt=fpath.read_text(encoding='utf-8')
            require(re.search(r'\b[A-Za-z]:[\\/]',txt) is None,
                    f'Local drive-letter path leaked in reviewer file: {fpath.name}')
            for tok in forbidden_tokens:
                require(tok not in txt,f'Forbidden outcome/source token {tok!r} in reviewer file: {fpath.name}')
    return {'schema':pub_schema,'invocation_token':tok_rev,'runner_sha256':rsha_rev,
            'n_provisional_clusters':n_clusters,'n_qa_panels':len(panel_names),
            'reviewer_files_verified':len(rev_man['files']),'private_files_verified':len(priv_man['files']),
            'both_trees_paired_and_verified':True}


def verify_published_adjudication_pair(dest,private_dest,expected_token=None,expected_runner_sha=None):
    rater_dir=Path(dest);priv_dir=Path(private_dest)
    require(rater_dir.is_dir(),f'Rater adjudication directory missing: {rater_dir}')
    require(priv_dir.is_dir(),f'Private adjudication outcomes directory missing: {priv_dir}')
    require(not Path(str(rater_dir)+'.staging').exists(),f'Stale rater staging directory exists: {rater_dir}.staging')
    require(not Path(str(priv_dir)+'.staging').exists(),f'Stale private staging directory exists: {priv_dir}.staging')
    rater_mpath=rater_dir/'SHA256_MANIFEST.json'
    priv_mpath=priv_dir/'SHA256_MANIFEST.json'
    require(rater_mpath.is_file() and priv_mpath.is_file(),'Both Stage 2 published trees must contain SHA256_MANIFEST.json')
    rater_man=json.loads(rater_mpath.read_text(encoding='utf-8'))
    priv_man=json.loads(priv_mpath.read_text(encoding='utf-8'))
    require(rater_man.get('schema')==SCHEMA and priv_man.get('schema')==SCHEMA,'Stage 2 published manifest schema mismatch')
    tok_rater=str(rater_man.get('invocation_token','')).strip()
    tok_priv=str(priv_man.get('invocation_token','')).strip()
    require(bool(tok_rater) and tok_rater==tok_priv,'Stage 2 published two-tree invocation_token mismatch or missing')
    if expected_token is not None:
        require(tok_rater==str(expected_token),'Stage 2 published invocation_token does not match active invocation')
    rsha_rater=str(rater_man.get('runner_sha256','')).strip()
    rsha_priv=str(priv_man.get('runner_sha256','')).strip()
    require(len(rsha_rater)==64 and rsha_rater==rsha_priv,'Stage 2 published two-tree runner_sha256 mismatch or missing')
    if expected_runner_sha is not None:
        require(rsha_rater==str(expected_runner_sha),'Stage 2 published runner_sha256 does not match expected runner SHA-256')
    for tree,man in ((rater_dir,rater_man),(priv_dir,priv_man)):
        expected_files=man.get('files',{})
        require(isinstance(expected_files,dict) and len(expected_files)>0,f'Empty file manifest in {tree}')
        actual_files={str(p.relative_to(tree)).replace('\\','/'):sha(p)
                      for p in sorted(tree.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
        require(set(actual_files.keys())==set(expected_files.keys()),
                f'On-disk file set differs from SHA256_MANIFEST.json in {tree}')
        for rel,digest in expected_files.items():
            require(actual_files[rel]==digest,f'File SHA-256 mismatch in {tree}: {rel}')
    blind_rep=json.loads((rater_dir/'stage2_blind_package_report.json').read_text(encoding='utf-8'))
    priv_rep=json.loads((priv_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
    sealed_readme=json.loads((priv_dir/'SEALED_STAGE2_README.json').read_text(encoding='utf-8'))
    require(blind_rep.get('schema')==SCHEMA and priv_rep.get('schema')==SCHEMA and sealed_readme.get('schema')==SCHEMA,
            'Stage 2 report/README schema mismatch')
    require(blind_rep.get('runner_sha256')==rsha_rater and priv_rep.get('runner_sha256')==rsha_rater
            and sealed_readme.get('runner_sha256')==rsha_rater,
            'Stage 2 report/README runner_sha256 mismatch with manifest')
    n_cases=int(blind_rep['blind_unique_object_cases'])
    require(n_cases>0 and int(priv_rep['blind_unique_object_cases'])==n_cases
            and int(sealed_readme['blind_unique_object_cases'])==n_cases,
            'Stage 2 rater report and private report blind case counts disagree')
    manifest_rows=read_csv(rater_dir/'blinded_case_manifest.csv')
    rating_rows=read_csv(rater_dir/'rating_form.csv')
    key_rows=read_csv(priv_dir/'SEALED_case_key_DO_NOT_SHARE_WITH_RATERS.csv')
    cids_man=[r['case_id'] for r in manifest_rows]
    cids_key=[r['case_id'] for r in key_rows]
    require(len(cids_man)==n_cases and cids_man==cids_key,
            'Stage 2 case_id sequence mismatch between blinded_case_manifest.csv and SEALED_case_key')
    require(len(rating_rows)==4*n_cases,'Stage 2 rating_form.csv must have exactly 4 rows per blind case')
    s1_pngs=sorted(p.name for p in (rater_dir/'stage1_blind').glob('*.png'))
    s2_pngs=sorted(p.name for p in (rater_dir/'stage2_blind').glob('*.png'))
    require(s1_pngs==sorted(f'{cid}.png' for cid in cids_man),
            'Stage 2 stage1_blind PNG panels do not match case IDs 1-to-1')
    require(s2_pngs==sorted([f'{cid}_A.png' for cid in cids_man]+[f'{cid}_B.png' for cid in cids_man]),
            'Stage 2 stage2_blind A/B PNG panels do not match case IDs 1-to-2')
    # Ensure zero sealed/outcome files or tokens leaked into rater-facing blind package
    for forbidden_file in ('SEALED_case_key_DO_NOT_SHARE_WITH_RATERS.csv','verified_objects.csv',
                           'per_object_persistence.csv','merged_object_persistence.csv',
                           'object_forensic_report.json','SEALED_STAGE2_README.json'):
        require(not (rater_dir/forbidden_file).exists(),
                f'Unblinded file leaked into rater package: {forbidden_file}')
    forbidden_tokens=('train_spatial_units.csv','train_spatial_diagnostic.json','train_fn_recoverability.json',
                      'SeaClips.zip','sampled_GT_is_tp','panel_A_semantics','panel_B_semantics',
                      'n_gt_fns','fn_frames','fn_fraction','persistent_80pct',
                      'is_tp','n_tp','n_fn','best_raw_query','OBJ_V','P00')
    for fpath in sorted(rater_dir.rglob('*')):
        if fpath.is_file() and fpath.suffix in ('.csv','.json','.md'):
            txt=fpath.read_text(encoding='utf-8')
            require(re.search(r'[A-Za-z]:[\\/]',txt) is None,
                    f'Local drive-letter path leaked in Stage 2 rater file: {fpath.name}')
            for tok in forbidden_tokens:
                require(tok not in txt,f'Forbidden outcome/key token {tok!r} in Stage 2 rater file: {fpath.name}')
        elif fpath.is_file() and fpath.suffix=='.png':
            from PIL import Image
            with Image.open(fpath) as pim:
                meta_str=json.dumps({str(k):str(v) for k,v in pim.info.items()})
                require(re.search(r'[A-Za-z]:[\\/]',meta_str) is None,
                        f'Local drive-letter path leaked in Stage 2 PNG metadata: {fpath.name}')
                for tok in forbidden_tokens:
                    require(tok not in meta_str,
                            f'Forbidden outcome/key token {tok!r} in Stage 2 PNG metadata: {fpath.name}')
    return {'schema':SCHEMA,'invocation_token':tok_rater,'runner_sha256':rsha_rater,
            'blind_unique_object_cases':n_cases,'stage1_blind_panels':len(s1_pngs),
            'stage2_blind_ab_panels':len(s2_pngs),
            'rater_files_verified':len(rater_man['files']),'private_files_verified':len(priv_man['files']),
            'both_stage2_trees_paired_and_verified':True}


def finish_output(tmp,dest):
    # All artifacts written and their SHA list captured before atomic directory rename.
    prepare_output_manifest(tmp)
    publish_staged_output(tmp,dest)


def sanitize_trace_snippet(snippet):
    s=str(snippet)
    s=re.sub(r'\b[A-Za-z]:[\\/][^\s\'"`)+,\];]*','<REDACTED_LOCAL_PATH>',s)
    s=re.sub(r'/(?:working_dir|tmp|home|root|Users)/[^\s\'"`)+,\];]*','<REDACTED_LOCAL_PATH>',s)
    for secret_name in ('seaclips-train.json','train_spatial_units.csv','train_spatial_diagnostic.json',
                        'train_fn_recoverability.json','SeaClips.zip','is_tp','n_tp','n_fn','best_raw_query',
                        'signed_center_x_over_gt_w','signed_center_y_over_gt_h',
                        'log_pred_w_over_gt_w','log_pred_h_over_gt_h'):
        s=re.sub(re.escape(secret_name),'<REDACTED_TOKEN>',s,flags=re.I)
    return s


def write_trace(folder,project):
    expected=['train_bfcef_f_s0_final_experts.py','run15a_final_windows.py',
              'run15a_final_mechanism.py','run15a_final_supervision.py',
              'run15a_internal_rebuild_PATCH03.py']
    lines=['# P-2588 frozen R50 input and O.r semantics trace (STATIC SOURCE INSPECTION ONLY)',
           '', 'No detector forward, tensor hook, import of project model, or training was performed.',
           'Any matches below are LOCATORS FOR HUMAN REVIEW, NOT verified transform semantics.',
           'Local filesystem paths and outcome identifiers are redacted before export.',
           'R50 effective input dimensions: UNVERIFIED.',
           'Color order / normalization / padding / mask / stride: UNVERIFIED.',
           'O.r semantics and common coordinate frame with GT: UNVERIFIED.',
           'Existing OCT13 cache showed O.r = ref_pts, but equality alone does not prove semantics.','']
    pat=re.compile(r'resize|padding|pad\(|normalize|interpolate|transform|stride|reference.points|decode.box|ref_pts|\bO\.r\b',re.I)
    for filename in expected:
        p=Path(project)/filename
        lines.extend(['## '+filename,''])
        if not p.is_file():lines.append('NOT AVAILABLE IN PROJECT DIRECTORY.\n');continue
        lines.append('Source SHA-256: `'+sha(p)+'`\n')
        text=p.read_text(encoding='utf-8',errors='replace').splitlines()
        loc=[(i+1,sanitize_trace_snippet(v.strip())) for i,v in enumerate(text) if pat.search(v)][:80]
        lines.append('Candidate source locations (max 80, not a semantic trace):')
        lines.extend([f'- line {i}: `{v[:170].replace("`", "’")}`' for i,v in loc])
        lines.append('')
    (folder/'source_decoder_trace.md').write_text('\n'.join(lines),encoding='utf-8')


def report_cluster_bounds(units):
    groups=defaultdict(list)
    for z in units:groups[(z['video'],z['class'])].append(z)
    out=[]
    for (vid,c),gg in sorted(groups.items()):
        frame_counts=Counter(z['frame'] for z in gg)
        out.append({'video':vid,'class':c,'class_name':CLASS_NAMES[c],
            'sampled_GT_rows':len(gg),
            'lower_bound_independent_objects_gt_only':max(frame_counts.values()),
            'upper_bound_independent_objects_gt_only':len(gg),
            'note':'Bounds only; no verified persistent identity'})
    return out


def render_image_crop(im,roi,max_width=460):
    from PIL import Image
    x0,y0,x1,y1=roi
    crop=im.crop((max(0,int(x0)),max(0,int(y0)),min(im.width,int(x1)),min(im.height,int(y1))))
    if crop.width and crop.height:
        k=min(max_width/max(1,crop.width),max_width/max(1,crop.height),5.0)
        if k>1:crop=crop.resize((round(crop.width*k),round(crop.height*k)),Image.Resampling.NEAREST)
    return crop


def qa_panel_sort_key(c):
    """Outcome-independent deterministic order by stable cluster ID and GT coordinates only."""
    cid=str(c['provisional_cluster_id']).strip()
    m_id=re.fullmatch(r'P(\d+)',cid)
    cid_key=(0,int(m_id.group(1)),cid) if m_id else (1,0,cid)
    first_aid=int(str(c['annotation_ids']).split(';')[0])
    return (cid_key,int(c['video']),int(c['class']),int(c.get('first_frame',0)),first_aid)


def order_qa_clusters(clusters):
    """Return clusters ordered deterministically without TP/FN or outcome-informed priorities."""
    require(len(clusters)>0,'Provisional cluster list must not be empty')
    cids=[str(c['provisional_cluster_id']).strip() for c in clusters]
    require(len(cids)==len(set(cids)) and all(bool(x) for x in cids),
            'Every provisional cluster must have a unique non-empty provisional_cluster_id')
    require(all(re.fullmatch(r'P\d{4}',cid) is not None and int(cid[1:])>=1 for cid in cids),
            'Malformed provisional_cluster_id; expected canonical P0001..P9999 format')
    return sorted(clusters,key=qa_panel_sort_key)


def identity_qa_panels(folder,clusters,units,images,image_root,max_panels=MAX_QA_PANELS_DEFAULT):
    """UNBLINDED linkage QA panels. Never substitute these for blinded detection review."""
    from PIL import Image,ImageDraw
    ordered=order_qa_clusters(clusters)
    require(max_panels is not None and int(max_panels)>=len(ordered),
            f'max_qa_panels ({max_panels}) must be >= cluster_count ({len(ordered)}); incomplete QA panel coverage forbidden')
    loc={z['annotation_id']:z for z in units}
    dst=folder/'identity_qa_panels';dst.mkdir(exist_ok=True)
    require(not list(dst.glob('*.png')),'QA panel destination already contains PNG files')
    rendered=0
    for c in ordered:
        anns=sorted((loc[int(i)] for i in c['annotation_ids'].split(';')),
                    key=lambda z:(int(z['frame']),int(z['annotation_id'])))
        require(len(anns)==int(c['n_observations']),'Cluster annotation count mismatch for '+str(c['provisional_cluster_id']))
        selected=[anns[0],anns[len(anns)//2],anns[-1]]
        pieces=[]
        for z in selected:
            im=load_source_rgb(image_root,images[z['image_id']])
            r=max(30,10*math.sqrt(z['area']))
            x0=max(0,int(z['cx']-r));y0=max(0,int(z['cy']-r))
            x1=min(im.width,int(z['cx']+r));y1=min(im.height,int(z['cy']+r))
            crop=render_image_crop(im,(x0,y0,x1,y1))
            dr=ImageDraw.Draw(crop); factor=crop.width/max(1,x1-x0)
            xx=(z['x']-x0)*factor;yy=(z['y']-y0)*factor
            dr.rectangle((xx,yy,xx+z['w']*factor,yy+z['h']*factor),outline='yellow',width=2)
            pieces.append(crop)
        width=sum(x.width for x in pieces)+20*(len(pieces)+1)
        height=max(x.height for x in pieces)+65
        board=Image.new('RGB',(width,height),'#252525');dr=ImageDraw.Draw(board)
        dr.text((12,8),f"LINKAGE QA ONLY: {c['provisional_cluster_id']}  video={c['video']} class={c['class']} UNVERIFIED",fill='white')
        offset=20
        for z,im in zip(selected,pieces):
            board.paste(im,(offset,45));offset+=im.width+20
        out_path=dst/(c['provisional_cluster_id']+'.png')
        require(not out_path.exists(),'Duplicate QA panel file: '+str(out_path))
        board.save(out_path)
        rendered+=1
    expected_files=[f"{c['provisional_cluster_id']}.png" for c in ordered]
    actual_files=sorted(p.name for p in dst.glob('*.png'))
    require(rendered==len(ordered) and actual_files==sorted(expected_files),
            f'Rendered QA panel count ({rendered}, files={len(actual_files)}) != cluster_count ({len(ordered)})')
    for fname in expected_files:
        with Image.open(dst/fname) as check_im:
            require(check_im.format=='PNG' and check_im.width>0 and check_im.height>0,
                    'Invalid rendered QA PNG panel: '+fname)
            check_im.load()
    return rendered


def build(args):
    require( getattr(args,'phase',None) in ('preflight','build'),
            'Unsupported build() phase: '+str(getattr(args,'phase',None)))
    if args.phase=='build':
        require(bool(getattr(args,'authorize_stage1_build',False)),
                'Stage 1 build is on HOLD; pass --authorize-stage1-build (authorize_stage1_build=True) only after explicit Project Director authorization')
    paths=source_paths(args.project,args.coco_train)
    pins=verify_sources(paths)
    rows,allow_imgs,videos=load_whitelist(paths['csv'],paths['spatial'],paths['fn'])
    raw_coco=json.loads(paths['coco'].read_text(encoding='utf-8'))
    imgs,anns=checked_authorized_coco(paths['coco'],rows,allow_imgs,raw=raw_coco)
    units,byimg=materialize_gt(rows,imgs,anns)
    source_ids=Counter()
    for aid,a in anns.items():
        for key,value in coco_track_key(a):source_ids[key]+=1
    edges,accepted,namb=provisional_edges(units)
    clusters,lookup=provisional_components(units,accepted)
    unitlookup={z['annotation_id']:z for z in units}
    for c in clusters:require(all(unitlookup[int(aid)]['video']==c['video'] for aid in c['annotation_ids'].split(';')),
                              'Cluster spanning videos')
    dest=Path(args.output)/'P2588_identity_build'
    ordered_clusters=order_qa_clusters(clusters)
    require(args.max_qa_panels>=len(ordered_clusters),
            f'max_qa_panels ({args.max_qa_panels}) must be >= cluster_count ({len(ordered_clusters)}) before writing outputs')
    private_dest=private_output_destination(args)
    require(args.images_root is None,'Extracted --images-root is not supported in ZIP-native v6')
    with TrainImagesZip(args.images_zip) as train_source:
        image_auth=verify_train_images_zip(train_source,imgs,allow_imgs)
        # No output side effects in preflight; archive is closed by context manager.
        if args.phase=='preflight':
            print('[P2588 EXACT SOURCE ID JOIN]',len(imgs),'GT-bearing images and',len(anns),'annotation IDs')
            print('[P2588 PROVISIONAL CLUSTERS]',len(ordered_clusters),'provisional clusters; max_qa_panels=',args.max_qa_panels)
            print('[TRAIN ZIP IMAGE HEADERS VERIFIED]',image_auth['n_verified_GT_bearing_source_image_headers'],
                  'COCO geometry and zip content integrity')
            ensure_output_unused(dest)
            ensure_output_unused(private_dest)
            print('[BOTH OUTPUTS UNUSED] True; reviewer/private trees separate')
            return
    # Re-open in the build phase; every QA-read byte string must equal the
    # SHA-256 validated during header preflight. Avoid re-reading other splits.
    header_image_hashes=dict(train_source.sha_by_id)
    # Both one-shot destinations must be absent BEFORE any output is written.
    ensure_output_unused(dest)
    ensure_output_unused(private_dest)
    print('[P2588 EXACT SOURCE ID JOIN]',len(imgs),'GT-bearing COCO image records and',
          len(anns),'annotation IDs verified; cache receipt',272,'sampled frames')
    print('[ZERO-GT FRAME STATUS]',24,'cache frames remain OUTSIDE object-level COCO identity study')
    print('[P2588 PREFLIGHT PASS]',len(units),'GT units /',
          len(ordered_clusters),'PROVISIONAL clusters (NOT physical-object ground truth)')
    print('[TRAIN ZIP IMAGE HEADERS VERIFIED]',image_auth['n_verified_GT_bearing_source_image_headers'],
          'exact GT-bearing source files, COCO dimensions checked')
    # Build both directories in a single protected transaction; stamp an invocation_token
    # at staging creation, finalize manifests for both staged trees before publishing either,
    # and roll back + verify removal of every directory owned by this invocation on any error.
    global ACTIVE_INVOCATION_TOKEN,ACTIVE_CREATED_STAGING,ACTIVE_PUBLISHED_FINAL,ACTIVE_FAILED_CLEANUP,ACTIVE_VERIFIED_ABSENT_STAGING
    invocation_token=uuid.uuid4().hex
    runner_sha256=current_runner_sha256()
    ACTIVE_INVOCATION_TOKEN=invocation_token
    created_staging=[]
    published_final=[]
    verified_absent_staging=[]
    ACTIVE_CREATED_STAGING=created_staging
    ACTIVE_PUBLISHED_FINAL=published_final
    ACTIVE_FAILED_CLEANUP=[]
    ACTIVE_VERIFIED_ABSENT_STAGING=verified_absent_staging
    candidate_paths=[dest,private_dest,Path(str(dest)+'.staging'),Path(str(private_dest)+'.staging')]
    try:
        tmp=tmp_output(dest)
        private_tmp=tmp_output(private_dest)
        write_csv(tmp/'gt_linkage_proposals.csv',edges,[
            'source_annotation_id','target_annotation_id','video','class','from_frame','to_frame',
            'displacement_px','displacement_over_gt_size','log_area_ratio_abs','gt_box_iou','cost',
            'second_cost_margin_from','second_cost_margin_to','linkage_status'])
        write_csv(tmp/'identity_qa_manifest.csv',ordered_clusters,[
            'provisional_cluster_id','video','class','class_name','n_observations',
            'first_frame','last_frame','annotation_ids','verification_status',
            'physical_object_id','identity_evidence','verified_by'])
        # Reviewer does not receive FN/TP outcomes: this is a GT-only identity decision.
        write_csv(tmp/'identity_review_template.csv',
                  [{**c,'verification_status':'UNREVIEWED'} for c in ordered_clusters],
                  ['provisional_cluster_id','video','class','class_name','n_observations',
                   'first_frame','last_frame','annotation_ids','verification_status',
                   'physical_object_id','identity_evidence','verified_by'])
        n=gt_neighbors(units,byimg)
        write_csv(tmp/'gt_neighbor_spacing.csv',n,list(n[0]))
        # Outcome counts and full local source paths are SEALED from identity reviewers.
        byaid={z['annotation_id']:z for z in units}
        outcomes=[{'provisional_cluster_id':c['provisional_cluster_id'],
                   'n_tp':sum(byaid[int(a)]['is_tp'] for a in c['annotation_ids'].split(';')),
                   'n_fn':sum(1-byaid[int(a)]['is_tp'] for a in c['annotation_ids'].split(';'))}
                  for c in ordered_clusters]
        write_csv(private_tmp/'SEALED_cluster_outcomes_DO_NOT_SHARE_WITH_IDENTITY_REVIEWERS.csv',outcomes,
                  ['provisional_cluster_id','n_tp','n_fn'])
        (private_tmp/'SEALED_README.json').write_text(json.dumps({
            'schema':SCHEMA,'runner_sha256':runner_sha256,'visibility':'PI_ONLY_DO_NOT_DISTRIBUTE',
            'source_pins':pins,'image_zip_path':str(image_auth.get('image_zip','')),
            'n_provisional_clusters':len(ordered_clusters),
            'note':('Physical separation is NOT OS-level access control. Never share this directory. '
                    'CSV outcomes and local source paths are not inputs to human identity verification.')},indent=2),encoding='utf-8')
        write_csv(tmp/'verified_objects.csv',[],['physical_object_id','video','class','verified_by','identity_evidence','n_observations'])
        write_csv(tmp/'per_object_persistence.csv',[],['physical_object_id','video','class','annotated_frames','fn_frames','fn_fraction'])
        write_csv(tmp/'reference_behavior.csv',[],['video','frame_from','frame_to','mean_ref_distance','max_ref_distance','status'])
        write_csv(tmp/'blinded_case_manifest.csv',[],['case_id','physical_object_id','status'])
        write_csv(tmp/'rating_form.csv',[],['case_id','rater_id','stage','object_visible','extent_matches','clutter','quality','notes'])
        write_trace(tmp,args.project)
        with TrainImagesZip(args.images_zip) as train_source:
            train_source.resolve(imgs,allow_imgs)
            train_source.sha_by_id=header_image_hashes
            rendered=identity_qa_panels(tmp,ordered_clusters,units,imgs,train_source,args.max_qa_panels)
        panel_files=sorted(p.name for p in (tmp/'identity_qa_panels').glob('*.png'))
        expected_panels=sorted(f"{c['provisional_cluster_id']}.png" for c in ordered_clusters)
        require(rendered==len(ordered_clusters) and panel_files==expected_panels,
                f'rendered_panel_count ({rendered}, files={len(panel_files)}) != cluster_count ({len(ordered_clusters)}); fail closed')
        # No source files, local filesystem paths, or image pixels included in reviewer report.
        public_image_auth={k:v for k,v in image_auth.items() if k not in ('image_root','image_zip')}
        public_image_auth['image_source_role']='PINNED_SEACLIPS_ARCHIVE_TRAIN_SPLIT'
        report={'schema':SCHEMA,'runner_sha256':runner_sha256,'status':'AWAITING_HUMAN_IDENTITY_VERIFICATION',
                'TRAIN_only':True,'no_training':True,'no_detector_forward':True,
                'source_pins':public_source_pins(pins),
                'n_frozen_cached_images':272,
                'n_gt_bearing_coco_image_records_source_id_verified':len(imgs),
                'n_gt_source_annotation_ids_verified':len(anns),
                'training_images_stay_inside_original_zip':True,
                'raw_source_categories':raw_coco['categories'],
                'train_image_header_verification':public_image_auth,
                'outcome_separation':'SEALED_OUTSIDE_REVIEWER_PACKAGE; SHARE_ONLY_P2588_identity_build',
                'n_zero_gt_cached_frames_NOT_resolved_to_coco':24,
                'coco_source_frame_offset_per_video':{str(v):sorted(set(
                    int(im['frame_id'])-int(im['_p2588_cached_frame_idx'])
                    for im in imgs.values() if int(im['video_id'])==v)) for v in TRAIN},
                'coco_coverage_note':'Only exact source image and annotation IDs from signed CSV; cached j is NOT COCO frame_id',
                'n_gt_units':len(units),'aggregate_TP_FNs_not_exported_to_identity_review':True,
                'n_provisional_clusters_NOT_independent_objects':len(ordered_clusters),
                'n_proposed_edges_NOT_verified':len(accepted),'n_ambiguous_edges':namb,
                'n_verified_physical_objects':0,'authoritative_source_track_keys_present':dict(source_ids),
                'gt_only_identity_bounds':report_cluster_bounds(units),
                'identity_QA_panels_rendered':rendered,
                'identity_QA_panel_coverage_complete':rendered==len(ordered_clusters),
                'identity_QA_panel_ordering':'OUTCOME_INDEPENDENT_PROVISIONAL_CLUSTER_ID_ASC',
                'effective_input_pixels':'UNVERIFIED',
                'O_r_semantics':'UNVERIFIED; no model forward or cache-reference load performed',
                'source_annotation_scope':'pinned official TRAIN JSON; GT-only exact 248-image and 1025-annotation allowlist; zero-GT cache frames are not source-resolved',
                'no_architecture_authorization':True,
                'limitations':['Image-stage panels are identity QA, NOT blinded object-detection assessments.',
                               'All association edges and clusters are provisional; human identity approval is required.',
                               'COCO annotation_id does not imply persistent identity.',
                               'GT center-distance is not proof of identity when camera moves or objects cross.',
                               'Physical-object sample size remains UNKNOWN until verified.',
                               'Authoritative preprocessing and O.r origin are not established by static keyword search.']}
        (tmp/'object_forensic_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        prepare_output_manifest(private_tmp,invocation_token=invocation_token)
        prepare_output_manifest(tmp,invocation_token=invocation_token)
        publish_staged_output(private_tmp,private_dest)
        publish_staged_output(tmp,dest)
        verify_published_build_pair(dest,private_dest,expected_token=invocation_token,expected_runner_sha=runner_sha256)
    except BaseException as build_exc:
        local_created=[p for p in (locals().get('tmp'),locals().get('private_tmp')) if isinstance(p,Path)]
        active_stg=list(ACTIVE_CREATED_STAGING) if isinstance(ACTIVE_CREATED_STAGING,list) else []
        active_pub=list(ACTIVE_PUBLISHED_FINAL) if isinstance(ACTIVE_PUBLISHED_FINAL,list) else []
        active_abs=list(ACTIVE_VERIFIED_ABSENT_STAGING) if isinstance(ACTIVE_VERIFIED_ABSENT_STAGING,list) else []
        owned_now=list(published_final)+active_pub+list(created_staging)+active_stg+local_created
        try:
            rollback_owned_outputs(
                list(reversed(published_final))+list(reversed(active_pub))+list(reversed(created_staging))+list(reversed(active_stg))+list(reversed(active_abs))+candidate_paths,
                invocation_token,
                owned_paths=owned_now,
                already_failed=ACTIVE_FAILED_CLEANUP,
                verified_absent_staging=list(verified_absent_staging)+active_abs,
            )
        except BaseException as rb_exc:
            if '[P2588 ROLLBACK FAILURE]' in str(build_exc):
                raise RuntimeError(f'{build_exc}; {rb_exc}') from (build_exc.__cause__ or build_exc)
            raise RuntimeError(f'{rb_exc} (while rolling back after: {build_exc})') from build_exc
        raise
    finally:
        ACTIVE_INVOCATION_TOKEN=None
        ACTIVE_CREATED_STAGING=None
        ACTIVE_PUBLISHED_FINAL=None
        ACTIVE_FAILED_CLEANUP=None
        ACTIVE_VERIFIED_ABSENT_STAGING=None
    print('[BUILD]',dest,'provisional_clusters',len(clusters),'identity_QA_images',rendered)
    print('[PRIVATE OUTCOMES — PI ONLY]',private_dest,'DO NOT SHARE WITH IDENTITY REVIEWERS')


CANDIDATE_MERGE_COLUMNS=(
    'video','class','class_name','earlier_cluster_id','later_cluster_id',
    'earlier_last_frame','later_first_frame','frame_gap',
    'visual_and_kinematic_rationale','merge_confidence'
)


def reviewer_attestation_mode(verified_by):
    who=str(verified_by or '').strip().lower()
    if who.startswith(('chatgpt','ai_','ai-','claude','gemini','llm')):
        return 'AI_BLIND_QA_ATTESTED'
    return 'HUMAN_ATTESTED'


def validate_human_decisions(manifest,review_rows):
    baseline={x['provisional_cluster_id']:x for x in manifest}
    seen=set();accepted=[];rejected=0
    for r in review_rows:
        cid=r['provisional_cluster_id'];require(cid in baseline,'Review names unknown cluster')
        require(cid not in seen,'Duplicate reviewed cluster');seen.add(cid)
        original=baseline[cid]
        for fld in ('video','class','class_name','n_observations','first_frame','last_frame','annotation_ids'):
            if fld in r and fld in original:
                require(str(r[fld])==str(original[fld]),'Tampered QA manifest '+cid+' '+fld)
        status=r['verification_status'].upper().strip()
        require(status in ('VERIFIED','REJECT','UNREVIEWED','UNRESOLVED'),'Unknown verification status')
        if status=='VERIFIED':
            oid=r['physical_object_id'].strip();who=r['verified_by'].strip();evidence=r['identity_evidence'].strip()
            require(oid and who and evidence in ('IMAGE_QA','AUTHORITATIVE_TRACK'),
                    'VERIFIED needs physical_object_id, verified_by and allowed evidence')
            require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',oid) is not None,'Unsafe physical object ID')
            accepted.append({**original,'physical_object_id':oid,'verified_by':who,
                             'identity_evidence':evidence,'verification_status':'HUMAN_ATTESTED',
                             'attestation_mode':reviewer_attestation_mode(who)})
        else:
            require(not str(r.get('physical_object_id','')).strip(),
                    'Non-VERIFIED cluster must not carry a physical_object_id: '+cid)
            rejected+=1
    require(set(baseline)==seen,'Reviewer must account for every provisional cluster')
    ids=[x['physical_object_id'] for x in accepted]
    require(len(ids)==len(set(ids)),
            'One physical ID may not silently merge distinct provisional clusters; separate audited merge protocol needed')
    return accepted,rejected


def validate_candidate_merges(manifest,verified,merge_rows,active_tiers=('HIGH','MEDIUM')):
    baseline={x['provisional_cluster_id']:x for x in manifest}
    verified_by_cid={x['provisional_cluster_id']:x for x in verified}
    active_set={str(t).upper().strip() for t in active_tiers}
    require(active_set.issubset({'HIGH','MEDIUM','LOW'}),'Invalid active merge confidence tiers')
    validated=[];seen_pairs=set()
    for idx,r in enumerate(merge_rows,1):
        for col in CANDIDATE_MERGE_COLUMNS:
            require(col in r,f'Candidate merge row {idx} missing column: {col}')
        ear_id=str(r['earlier_cluster_id']).strip()
        lat_id=str(r['later_cluster_id']).strip()
        require(ear_id in baseline and lat_id in baseline,
                f'Candidate merge row {idx} references unknown cluster: {ear_id} -> {lat_id}')
        require(ear_id!=lat_id,f'Candidate merge row {idx} self-merge forbidden: {ear_id}')
        require((ear_id,lat_id) not in seen_pairs and (lat_id,ear_id) not in seen_pairs,
                f'Duplicate candidate merge pair: {ear_id} -> {lat_id}')
        seen_pairs.add((ear_id,lat_id))
        ear=baseline[ear_id];lat=baseline[lat_id]
        vid=int(r['video']);cl=int(r['class']);cname=str(r['class_name']).strip()
        require(int(ear['video'])==vid and int(lat['video'])==vid,
                f'Candidate merge video mismatch on {ear_id} -> {lat_id}')
        require(int(ear['class'])==cl and int(lat['class'])==cl and CLASS_NAMES[cl]==cname,
                f'Candidate merge class mismatch on {ear_id} -> {lat_id}')
        ear_last=int(r['earlier_last_frame']);lat_first=int(r['later_first_frame']);gap=int(r['frame_gap'])
        require(int(ear['last_frame'])==ear_last and int(lat['first_frame'])==lat_first,
                f'Candidate merge frame boundary mismatch on {ear_id} -> {lat_id}')
        require(ear_last<lat_first and gap==lat_first-ear_last and gap>=10 and gap%10==0,
                f'Candidate merge invalid frame_gap={gap} on {ear_id} -> {lat_id}')
        conf=str(r['merge_confidence']).upper().strip()
        require(conf in ('HIGH','MEDIUM','LOW'),f'Invalid merge_confidence {conf!r} on {ear_id} -> {lat_id}')
        rationale=str(r['visual_and_kinematic_rationale']).strip()
        require(bool(rationale),f'Empty visual_and_kinematic_rationale on {ear_id} -> {lat_id}')
        validated.append({
            'video':vid,'class':cl,'class_name':cname,
            'earlier_cluster_id':ear_id,'later_cluster_id':lat_id,
            'earlier_last_frame':ear_last,'later_first_frame':lat_first,
            'frame_gap':gap,'visual_and_kinematic_rationale':rationale,
            'merge_confidence':conf
        })
    parent={cid:cid for cid in verified_by_cid}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for mrow in validated:
        if mrow['merge_confidence'] not in active_set:
            continue
        a=mrow['earlier_cluster_id'];b=mrow['later_cluster_id']
        if a in parent and b in parent:
            ra=root(a);rb=root(b)
            if ra!=rb:
                if qa_panel_sort_key(verified_by_cid[ra])<=qa_panel_sort_key(verified_by_cid[rb]):
                    parent[rb]=ra
                else:
                    parent[ra]=rb
    grouped=defaultdict(list)
    for cid in sorted(verified_by_cid,key=lambda k:qa_panel_sort_key(verified_by_cid[k])):
        grouped[root(cid)].append(verified_by_cid[cid])
    comp_by_cid={};merged_components=[]
    for rcid,members in sorted(grouped.items(),key=lambda kv:qa_panel_sort_key(kv[1][0])):
        members_sorted=sorted(members,key=lambda c:(int(c['first_frame']),int(c['last_frame']),c['provisional_cluster_id']))
        vid=int(members_sorted[0]['video']);cl=int(members_sorted[0]['class'])
        comp_id=f"MCOMP_V{vid:02d}_C{cl}_{members_sorted[0]['provisional_cluster_id']}"
        all_aids=[]
        for m_c in members_sorted:
            comp_by_cid[m_c['provisional_cluster_id']]=comp_id
            all_aids.extend(int(x) for x in str(m_c['annotation_ids']).split(';'))
        require(len(all_aids)==len(set(all_aids)),f'Duplicate annotation_id in merge component {comp_id}')
        merged_components.append({
            'merge_component_id':comp_id,
            'primary_provisional_cluster_id':members_sorted[0]['provisional_cluster_id'],
            'primary_physical_object_id':members_sorted[0]['physical_object_id'],
            'video':vid,'class':cl,'class_name':CLASS_NAMES[cl],
            'n_merged_clusters':len(members_sorted),
            'member_cluster_ids':';'.join(c['provisional_cluster_id'] for c in members_sorted),
            'member_physical_object_ids':';'.join(c['physical_object_id'] for c in members_sorted),
            'annotation_ids':';'.join(str(x) for x in all_aids),
            'merge_confidence_tier':'+'.join(sorted(active_set)) if len(members_sorted)>1 else 'UNMERGED_SINGLE_CLUSTER',
        })
    return validated,comp_by_cid,merged_components


def sampling_stratum_for_cluster(c):
    v=int(c['video']);cl=int(c['class'])
    return (0 if (v==18 and cl==2) else 1 if (v==18 and cl==0) else
            2 if (v==4 and cl==2) else 3 if (v==24 and cl==3) else
            4 if (v in (7,20,25,35) and int(c.get('n_fn',0))==0) else 5)


def select_blind_cases(verified,seed=SEED,k=MAX_BLIND_CASES,merge_component_by_cid=None):
    rng=random.Random(seed)
    grouped=defaultdict(list)
    for c in verified:
        grouped[sampling_stratum_for_cluster(c)].append(c)
    quotas={0:12,1:6,2:6,3:6,4:8,5:2}
    def comp_key(c):
        cid=c['provisional_cluster_id']
        if merge_component_by_cid and cid in merge_component_by_cid:
            return merge_component_by_cid[cid]
        if c.get('merge_component_id'):
            return c['merge_component_id']
        return c['physical_object_id']
    chosen=[];taken=set();taken_comps=set()
    for grp in sorted(quotas):
        pool=sorted(grouped.get(grp,[]),key=lambda c:c['provisional_cluster_id']);rng.shuffle(pool)
        grp_chosen=[]
        # Pass 1: select distinct physical merge components first
        for x in pool:
            if len(grp_chosen)>=quotas[grp]:break
            ck=comp_key(x)
            if ck not in taken_comps:
                grp_chosen.append(x);taken.add(x['physical_object_id']);taken_comps.add(ck)
        # Pass 2: if stratum has fewer merge components than quota (e.g., Group 2 v4_c2), backfill within stratum
        for x in pool:
            if len(grp_chosen)>=quotas[grp]:break
            if x['physical_object_id'] not in taken:
                grp_chosen.append(x);taken.add(x['physical_object_id'])
        chosen.extend(grp_chosen)
    rest=sorted((x for x in verified if x['physical_object_id'] not in taken),
                key=lambda c:c['provisional_cluster_id'])
    rng.shuffle(rest)
    for x in rest:
        if len(chosen)>=k:break
        ck=comp_key(x)
        if ck not in taken_comps:
            chosen.append(x);taken.add(x['physical_object_id']);taken_comps.add(ck)
    for x in rest:
        if len(chosen)>=k:break
        if x['physical_object_id'] not in taken:
            chosen.append(x);taken.add(x['physical_object_id'])
    return chosen[:k]


def oracle_box_from_unit(z):
    """Reconstruct GT-SELECTED ORACLE query geometry, NEVER label native score output."""
    return {'x':z['cx']+z['signed_center_x_over_gt_w']*z['w']-z['w']*math.exp(z['log_pred_w_over_gt_w'])/2,
            'y':z['cy']+z['signed_center_y_over_gt_h']*z['h']-z['h']*math.exp(z['log_pred_h_over_gt_h'])/2,
            'w':z['w']*math.exp(z['log_pred_w_over_gt_w']),
            'h':z['h']*math.exp(z['log_pred_h_over_gt_h'])}


def render_blind_panels(folder,cases,units,images,image_source,seed=SEED):
    from PIL import Image,ImageDraw
    unit_by_id={z['annotation_id']:z for z in units}
    rand=random.Random(seed)
    one=folder/'stage1_blind';two=folder/'stage2_blind'
    one.mkdir(exist_ok=True);two.mkdir(exist_ok=True)
    require(not list(one.glob('*.png')) and not list(two.glob('*.png')),
            'Stage 2 blind panel directories must be empty before rendering')
    sealed=[];review=[]
    seen_comp=set()
    for i,c in enumerate(cases,1):
        anns=[unit_by_id[int(v)] for v in c['annotation_ids'].split(';')]
        anns=sorted(anns,key=lambda z:(z['frame'],z['annotation_id']))
        z=anns[(len(anns)-1)//2] # median frame, outcome-independent
        src=load_source_rgb(image_source,images[z['image_id']])
        if isinstance(image_source,TrainImagesZip):
            locator=f"{image_source.path}::{image_source.image_members[int(z['image_id'])]}"
        else:
            locator=str(image_path(image_source,images[z['image_id']]['file_name']))
        r=max(64,12*math.sqrt(z['area']))
        x0=max(0,int(z['cx']-r));y0=max(0,int(z['cy']-r))
        x1=min(src.width,int(z['cx']+r));y1=min(src.height,int(z['cy']+r))
        # stage 1 contains only pixels, context and a neutral scale bar; no annotation status
        base=src.crop((x0,y0,x1,y1))
        factor=max(1,min(6,int(480/max(base.width,base.height))))
        zoom=base.resize((base.width*factor,base.height*factor),Image.Resampling.NEAREST)
        case=f'CASE_{i:03d}';plate=Image.new('RGB',(max(500,zoom.width+24),zoom.height+70),'#181818')
        plate.paste(zoom,(12,42));dr=ImageDraw.Draw(plate)
        dr.text((12,12),f'{case}  original pixels x{factor}  // no enhancement',fill='white')
        # 10 source-pixel scale bar, deliberately does NOT point to GT
        dr.line((18,zoom.height+55,18+10*factor,zoom.height+55),fill='white',width=2)
        dr.text((25+10*factor,zoom.height+48),'10 source px',fill='white')
        s1_path=one/(case+'.png')
        plate.save(s1_path)
        with Image.open(s1_path) as check_im:
            require(check_im.format=='PNG' and check_im.width>0 and check_im.height>0,
                    'Invalid Stage 2 stage1_blind PNG: '+case)
            check_im.load()
        # The B panel is GT-selected best-IoU query (oracle); not score-ranked native output.
        boxes={'GT':z,'ORACLE':oracle_box_from_unit(z)}
        keys=['GT','ORACLE'];rand.shuffle(keys);mapping={letter:kind for letter,kind in zip(('A','B'),keys)}
        for label,kind in mapping.items():
            bd=zoom.copy();draw=ImageDraw.Draw(bd);b=boxes[kind]
            xx=(b['x']-x0)*factor;yy=(b['y']-y0)*factor
            draw.rectangle((xx,yy,xx+b['w']*factor,yy+b['h']*factor),outline='white',width=2)
            cover=Image.new('RGB',(max(500,bd.width+24),bd.height+45),'#181818')
            cover.paste(bd,(12,34));ImageDraw.Draw(cover).text((12,8),f'{case} // Panel {label}',fill='white')
            s2_path=two/(f'{case}_{label}.png')
            cover.save(s2_path)
            with Image.open(s2_path) as check_im:
                require(check_im.format=='PNG' and check_im.width>0 and check_im.height>0,
                        f'Invalid Stage 2 stage2_blind PNG: {case}_{label}')
                check_im.load()
        mcomp=str(c.get('merge_component_id') or c['physical_object_id'])
        is_sec=mcomp in seen_comp
        seen_comp.add(mcomp)
        sealed.append({'case_id':case,'physical_object_id':c['physical_object_id'],
                       'provisional_cluster_id':c.get('provisional_cluster_id',''),
                       'merge_component_id':mcomp,
                       'is_secondary_merge_fragment':is_sec,
                       'sampling_stratum':sampling_stratum_for_cluster({**z,**c}),
                       'video':z['video'],'class':z['class'],'frame':z['frame'],'image_id':z['image_id'],
                       'annotation_id':z['annotation_id'],'sampled_GT_is_tp_DESCRIPTIVE':z['is_tp'],
                       'image_path':locator,'panel_A_semantics':mapping['A'],
                       'panel_B_semantics':mapping['B'],
                       'native_score_ranked_prediction':'NOT_AVAILABLE_NO_NEW_FORWARD',
                       'oracle_note':'GT-best-IoU query is outcome-selected; comparison is descriptive'})
        for rater in (1,2):
            for stage in (1,2):
                review.append({'case_id':case,'rater_id':f'RATER_{rater}','stage':stage,
                               'object_visible':'','extent_matches':'','clutter':'','quality':'','notes':''})
    return sealed,review


def adjudicate(args):
    require(getattr(args,'phase',None) in ('review-preflight','adjudicate'),
            'Stage 2 human adjudication on HOLD; unsupported adjudicate() phase: '+str(getattr(args,'phase',None)))
    if args.phase=='adjudicate':
        require(bool(getattr(args,'authorize_stage2_adjudicate',False)),
                'Stage 2 human adjudication on HOLD; pass --authorize-stage2-adjudicate (authorize_stage2_adjudicate=True) only after explicit Project Director authorization')
    require(getattr(args,'identity_review',None) is not None and str(args.identity_review).strip(),
            '--identity-review path is required for review-preflight and adjudicate')
    review_path=Path(args.identity_review).resolve(strict=True)
    require(review_path.is_file(),f'Identity review CSV not found: {review_path}')
    path=source_paths(args.project,args.coco_train);pins=verify_sources(path)
    units_csv,allow_imgs,_=load_whitelist(path['csv'],path['spatial'],path['fn'])
    imgs,anns=checked_authorized_coco(path['coco'],units_csv,allow_imgs)
    units,byimg=materialize_gt(units_csv,imgs,anns)
    build_dir=Path(args.output)/'P2588_identity_build'
    private_build_dir=private_output_destination(args)
    stage1_pair=verify_published_build_pair(
        build_dir,private_build_dir,expected_runner_sha=current_runner_sha256(),allow_pinned_stage1_v11=True
    )
    report=json.loads((build_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
    require(report['source_pins']==public_source_pins(pins) and report['status']=='AWAITING_HUMAN_IDENTITY_VERIFICATION',
            'Build source/status changed')
    original=read_csv(build_dir/'identity_qa_manifest.csv')
    decisions=read_csv(review_path)
    verified,n_rejected=validate_human_decisions(original,decisions)
    require(len(verified)>0,'At least one VERIFIED cluster is required for Stage 2 adjudication')
    merges_arg=getattr(args,'candidate_merges',None)
    merges_sha=None
    if merges_arg is not None and str(merges_arg).strip():
        merges_path=Path(merges_arg).resolve(strict=True)
        require(merges_path.is_file(),f'Candidate merges CSV not found: {merges_path}')
        merges_sha=sha(merges_path)
        raw_merges=read_csv(merges_path)
    else:
        raw_merges=[]
    validated_merges,comp_by_cid,merged_components=validate_candidate_merges(original,verified,raw_merges)
    byaid={z['annotation_id']:z for z in units}
    for c in verified:
        a=[byaid[int(s)] for s in c['annotation_ids'].split(';')]
        c['n_tp']=sum(z['is_tp'] for z in a)
        c['n_fn']=sum(1-z['is_tp'] for z in a)
        c['merge_component_id']=comp_by_cid.get(c['provisional_cluster_id'],c['physical_object_id'])
        # Ensure frame uniqueness inside every verified cluster
        frames=[z['frame'] for z in a]
        require(len(frames)==len(set(frames)),'Duplicate frame in verified cluster '+str(c['provisional_cluster_id']))
    for mcomp in merged_components:
        a=[byaid[int(s)] for s in mcomp['annotation_ids'].split(';')]
        frames=[z['frame'] for z in a]
        require(len(frames)==len(set(frames)),
                'Duplicate frame across merged verified clusters in '+str(mcomp['merge_component_id']))
        n_obs=len(a);n_tp=sum(z['is_tp'] for z in a);n_fn=sum(1-z['is_tp'] for z in a)
        mcomp['annotated_frames']=n_obs
        mcomp['tp_frames']=n_tp
        mcomp['fn_frames']=n_fn
        mcomp['fn_fraction']=n_fn/n_obs
        mcomp['persistent_80pct_descriptive']=(n_fn/n_obs)>=0.8
    cases=select_blind_cases(verified,seed=SEED,k=MAX_BLIND_CASES,merge_component_by_cid=comp_by_cid)
    require(len(cases)==min(MAX_BLIND_CASES,len(verified)),'Unexpected blind case selection count')
    stratum_ver_counts=Counter(sampling_stratum_for_cluster(c) for c in verified)
    stratum_case_counts=Counter(sampling_stratum_for_cluster(c) for c in cases)
    dest=Path(args.output)/'P2588_adjudication'
    private_adj_dest=private_adjudication_destination(args)
    require(args.images_root is None,'Extracted --images-root is not supported in ZIP-native Stage 2 v12')
    with TrainImagesZip(args.images_zip) as train_source:
        image_auth=verify_train_images_zip(train_source,imgs,allow_imgs)
        if args.phase=='review-preflight':
            print('[P2588 STAGE 1 BUILD PAIR VERIFIED] token=',stage1_pair['invocation_token'],'schema=',stage1_pair['schema'])
            print('[ADJUDICATE PREFLIGHT]',len(verified),'verified clusters;',n_rejected,'unresolved/rejected;',
                  len(merged_components),'merge-consolidated physical objects')
            print('[STAGE 2 STRATUM VERIFIED COUNTS]',dict(sorted(stratum_ver_counts.items())))
            print('[STAGE 2 STRATUM SAMPLED CASE COUNTS]',dict(sorted(stratum_case_counts.items())),'total_cases=',len(cases))
            print('[TRAIN ZIP IMAGE HEADERS VERIFIED]',image_auth['n_verified_GT_bearing_source_image_headers'])
            ensure_output_unused(dest)
            ensure_output_unused(private_adj_dest)
            print('[BOTH STAGE 2 OUTPUTS UNUSED] True; rater/private trees separate')
            return
    header_image_hashes=dict(train_source.sha_by_id)
    ensure_output_unused(dest)
    ensure_output_unused(private_adj_dest)
    vidclass=Counter((int(c['video']),int(c['class'])) for c in verified)
    vidclass_merged=Counter((int(m['video']),int(m['class'])) for m in merged_components)
    global ACTIVE_INVOCATION_TOKEN,ACTIVE_CREATED_STAGING,ACTIVE_PUBLISHED_FINAL,ACTIVE_FAILED_CLEANUP,ACTIVE_VERIFIED_ABSENT_STAGING
    invocation_token=uuid.uuid4().hex
    runner_sha256=current_runner_sha256()
    ACTIVE_INVOCATION_TOKEN=invocation_token
    created_staging=[]
    published_final=[]
    verified_absent_staging=[]
    ACTIVE_CREATED_STAGING=created_staging
    ACTIVE_PUBLISHED_FINAL=published_final
    ACTIVE_FAILED_CLEANUP=[]
    ACTIVE_VERIFIED_ABSENT_STAGING=verified_absent_staging
    candidate_paths=[dest,private_adj_dest,Path(str(dest)+'.staging'),Path(str(private_adj_dest)+'.staging')]
    try:
        tmp=tmp_output(dest)
        private_tmp=tmp_output(private_adj_dest)
        accepted=[];per=[]
        for c in verified:
            accepted.append({'physical_object_id':c['physical_object_id'],
                             'provisional_cluster_id':c['provisional_cluster_id'],
                             'merge_component_id':c['merge_component_id'],
                             'video':c['video'],'class':c['class'],
                             'verified_by':c['verified_by'],
                             'attestation_mode':c.get('attestation_mode',reviewer_attestation_mode(c['verified_by'])),
                             'identity_evidence':c['identity_evidence'],
                             'n_observations':c['n_observations'],
                             'n_gt_fns':c['n_fn'],'annotation_ids':c['annotation_ids']})
            n=int(c['n_observations']);nf=int(c['n_fn'])
            per.append({'physical_object_id':c['physical_object_id'],
                        'provisional_cluster_id':c['provisional_cluster_id'],
                        'merge_component_id':c['merge_component_id'],
                        'video':c['video'],'class':c['class'],
                        'annotated_frames':n,'fn_frames':nf,'fn_fraction':nf/n,
                        'persistent_80pct_descriptive':nf/n>=.8})
        # Outcome-bearing tables go ONLY to private_tmp, never to rater-facing tmp
        write_csv(private_tmp/'verified_objects.csv',accepted,[
            'physical_object_id','provisional_cluster_id','merge_component_id','video','class',
            'verified_by','attestation_mode','identity_evidence','n_observations','n_gt_fns','annotation_ids'])
        write_csv(private_tmp/'per_object_persistence.csv',per,[
            'physical_object_id','provisional_cluster_id','merge_component_id','video','class',
            'annotated_frames','fn_frames','fn_fraction','persistent_80pct_descriptive'])
        write_csv(private_tmp/'merged_object_persistence.csv',merged_components,[
            'merge_component_id','primary_provisional_cluster_id','primary_physical_object_id',
            'video','class','class_name','n_merged_clusters','member_cluster_ids',
            'member_physical_object_ids','annotated_frames','tp_frames','fn_frames',
            'fn_fraction','persistent_80pct_descriptive','merge_confidence_tier','annotation_ids'])
        write_csv(private_tmp/'candidate_cluster_merges_validated.csv',validated_merges,list(CANDIDATE_MERGE_COLUMNS))
        with TrainImagesZip(args.images_zip) as train_source:
            train_source.resolve(imgs,allow_imgs)
            train_source.sha_by_id=header_image_hashes
            sealed,form=render_blind_panels(tmp,cases,units,imgs,train_source)
        require(len(sealed)==len(cases) and len(form)==4*len(cases),
                'Stage 2 blind panel rendering incomplete; fail closed')
        write_csv(tmp/'blinded_case_manifest.csv',[{'case_id':z['case_id'],'status':'BLIND_TO_RATERS',
            'panel_stage1':'stage1_blind/'+z['case_id']+'.png',
            'panel_A':'stage2_blind/'+z['case_id']+'_A.png',
            'panel_B':'stage2_blind/'+z['case_id']+'_B.png'} for z in sealed],
            ['case_id','status','panel_stage1','panel_A','panel_B'])
        write_csv(private_tmp/'SEALED_case_key_DO_NOT_SHARE_WITH_RATERS.csv',sealed,[
            'case_id','physical_object_id','provisional_cluster_id','merge_component_id',
            'is_secondary_merge_fragment','sampling_stratum','video','class','frame','image_id',
            'annotation_id','sampled_GT_is_tp_DESCRIPTIVE','image_path','panel_A_semantics',
            'panel_B_semantics','native_score_ranked_prediction','oracle_note'])
        write_csv(tmp/'rating_form.csv',form,['case_id','rater_id','stage','object_visible','extent_matches',
                                             'clutter','quality','notes'])
        (tmp/'source_decoder_trace.md').write_bytes((build_dir/'source_decoder_trace.md').read_bytes())
        write_csv(tmp/'reference_behavior.csv',[],['video','frame_from','frame_to','mean_ref_distance',
                                                    'max_ref_distance','status'])
        public_image_auth={k:v for k,v in image_auth.items() if k not in ('image_root','image_zip')}
        public_image_auth['image_source_role']='PINNED_SEACLIPS_ARCHIVE_TRAIN_SPLIT'
        attestation_modes=sorted(set(x['attestation_mode'] for x in accepted))
        blind_report={
            'schema':SCHEMA,'runner_sha256':runner_sha256,
            'stage1_build_schema':stage1_pair['schema'],
            'stage1_invocation_token':stage1_pair['invocation_token'],
            'status':'RATER_REVIEW_PENDING',
            'TRAIN_only':True,'no_training':True,'no_detector_forward':True,
            'source_pins':public_source_pins(pins),
            'train_image_header_verification':public_image_auth,
            'identity_review_sha256':sha(review_path),
            'candidate_merges_sha256':merges_sha,
            'stage1_attestation_modes':attestation_modes,
            'blind_unique_object_cases':len(sealed),
            'stage1_identity_hidden':True,'stage2_A_B_randomized':True,
            'stage2_oracle_is_GT_SELECTED_NOT_NATIVE':True,
            'outcome_and_key_separation':'SEALED_OUTSIDE_RATER_PACKAGE; SHARE_ONLY_P2588_adjudication',
            'two_rater_rubric_status':'FORMS_PREPARED_NO_RATINGS',
            'input_tensor_transform':'UNVERIFIED','O_r_semantics':'UNVERIFIED',
            'no_architecture_authorization':True,
        }
        (tmp/'stage2_blind_package_report.json').write_text(
            json.dumps(blind_report,indent=2,allow_nan=False),encoding='utf-8'
        )
        n_unique_mcomp_in_cases=len(set(z['merge_component_id'] for z in sealed))
        priv_out={
            'schema':SCHEMA,'runner_sha256':runner_sha256,
            'visibility':'PI_ONLY_DO_NOT_SHARE_WITH_RATERS',
            'stage1_build_schema':stage1_pair['schema'],
            'stage1_invocation_token':stage1_pair['invocation_token'],
            'status':'RATER_REVIEW_PENDING',
            'source_pins':pins,
            'identity_review_sha256':sha(review_path),
            'candidate_merges_sha256':merges_sha,
            'stage1_attestation_modes':attestation_modes,
            'human_attested_distinct_object_ids':len(accepted),
            'n_unresolved_or_rejected_clusters':n_rejected,
            'n_candidate_merge_pairs_validated':len(validated_merges),
            'n_merge_consolidated_physical_objects':len(merged_components),
            'not_independent_events':sum(int(c['n_observations']) for c in verified),
            'n_persistent_ge80pct_descriptive':sum(int(x['persistent_80pct_descriptive']) for x in per),
            'n_merged_persistent_ge80pct_descriptive':sum(int(x['persistent_80pct_descriptive']) for x in merged_components),
            'class_video_counts':[{'video':v,'class':c,'verified_objects':n,
                                   'merged_physical_objects':vidclass_merged[(v,c)]}
                                  for (v,c),n in sorted(vidclass.items())],
            'stratum_verified_cluster_counts':{str(k):v for k,v in sorted(stratum_ver_counts.items())},
            'stratum_sampled_case_counts':{str(k):v for k,v in sorted(stratum_case_counts.items())},
            'blind_unique_object_cases':len(sealed),
            'blind_unique_merge_components_sampled':n_unique_mcomp_in_cases,
            'stage1_identity_hidden':True,'stage2_A_B_randomized':True,
            'stage2_oracle_is_GT_SELECTED_NOT_NATIVE':True,
            'two_rater_rubric_status':'FORMS_PREPARED_NO_RATINGS',
            'input_tensor_transform':'UNVERIFIED','O_r_semantics':'UNVERIFIED',
            'H_S1':'INDETERMINATE','H_S2':'INDETERMINATE','H_S4':'INDETERMINATE','H_S5':'INDETERMINATE',
            'no_architecture_authorization':True,
        }
        (private_tmp/'object_forensic_report.json').write_text(
            json.dumps(priv_out,indent=2,allow_nan=False),encoding='utf-8'
        )
        (private_tmp/'SEALED_STAGE2_README.json').write_text(json.dumps({
            'schema':SCHEMA,'runner_sha256':runner_sha256,
            'visibility':'PI_ONLY_DO_NOT_DISTRIBUTE_TO_STAGE2_RATERS',
            'stage1_invocation_token':stage1_pair['invocation_token'],
            'source_pins':pins,'image_zip_path':str(image_auth.get('image_zip','')),
            'identity_review_path':str(review_path),'identity_review_sha256':sha(review_path),
            'candidate_merges_sha256':merges_sha,
            'blind_unique_object_cases':len(sealed),
            'note':('Contains SEALED_case_key_DO_NOT_SHARE_WITH_RATERS.csv (A/B GT vs ORACLE key) '
                    'and TP/FN persistence tables. Share ONLY P2588_adjudication with Stage 2 raters.')
        },indent=2,allow_nan=False),encoding='utf-8')
        prepare_output_manifest(private_tmp,invocation_token=invocation_token)
        prepare_output_manifest(tmp,invocation_token=invocation_token)
        publish_staged_output(private_tmp,private_adj_dest)
        publish_staged_output(tmp,dest)
        verify_published_adjudication_pair(
            dest,private_adj_dest,expected_token=invocation_token,expected_runner_sha=runner_sha256
        )
    except BaseException as adj_exc:
        local_created=[p for p in (locals().get('tmp'),locals().get('private_tmp')) if isinstance(p,Path)]
        active_stg=list(ACTIVE_CREATED_STAGING) if isinstance(ACTIVE_CREATED_STAGING,list) else []
        active_pub=list(ACTIVE_PUBLISHED_FINAL) if isinstance(ACTIVE_PUBLISHED_FINAL,list) else []
        active_abs=list(ACTIVE_VERIFIED_ABSENT_STAGING) if isinstance(ACTIVE_VERIFIED_ABSENT_STAGING,list) else []
        owned_now=list(published_final)+active_pub+list(created_staging)+active_stg+local_created
        try:
            rollback_owned_outputs(
                list(reversed(published_final))+list(reversed(active_pub))+list(reversed(created_staging))+list(reversed(active_stg))+list(reversed(active_abs))+candidate_paths,
                invocation_token,
                owned_paths=owned_now,
                already_failed=ACTIVE_FAILED_CLEANUP,
                verified_absent_staging=list(verified_absent_staging)+active_abs,
            )
        except BaseException as rb_exc:
            if '[P2588 ROLLBACK FAILURE]' in str(adj_exc):
                raise RuntimeError(f'{adj_exc}; {rb_exc}') from (adj_exc.__cause__ or adj_exc)
            raise RuntimeError(f'{rb_exc} (while rolling back Stage 2 after: {adj_exc})') from adj_exc
        raise
    finally:
        ACTIVE_INVOCATION_TOKEN=None
        ACTIVE_CREATED_STAGING=None
        ACTIVE_PUBLISHED_FINAL=None
        ACTIVE_FAILED_CLEANUP=None
        ACTIVE_VERIFIED_ABSENT_STAGING=None
    print('[ADJUDICATE]',priv_out['status'],'verified_objects',len(accepted),
          'merged_physical_objects',len(merged_components),
          'blind_objects',len(sealed),'rater_path',dest)
    print('[PRIVATE STAGE 2 OUTCOMES & KEY — PI ONLY]',private_adj_dest,'DO NOT SHARE WITH RATERS')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',required=True,choices=('selftest','inventory','zip-inventory','preflight','build','verify-build','review-preflight','adjudicate','verify-adjudication'))
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    ap.add_argument('--coco-train',default=r'D:\MARIVISION\seaclips-train.json')
    ap.add_argument('--output',default=None,help='Dedicated NEW output parent; never a historic receipt directory')
    ap.add_argument('--images-root',default=None,help='Deprecated; refuse extracted directory in v5')
    ap.add_argument('--images-zip',default=r'D:\MARIVISION\SeaClips.zip',help='Original SeaClips archive; never extract full dataset')
    ap.add_argument('--sealed-root',default=None,help='Dedicated private root OUTSIDE reviewer output tree; required for preflight/build/adjudicate')
    ap.add_argument('--max-qa-panels',type=int,default=MAX_QA_PANELS_DEFAULT)
    ap.add_argument('--identity-review',default=None,help='Completed Stage 1 identity_review_completed.csv')
    ap.add_argument('--candidate-merges',default=None,help='Optional audited candidate_cluster_merges.csv')
    ap.add_argument('--authorize-stage1-build',action='store_true',default=False,
                    help='Explicit Project Director authorization required to execute --phase build')
    ap.add_argument('--authorize-stage2-adjudicate',action='store_true',default=False,
                    help='Explicit Project Director authorization required to execute --phase adjudicate')
    a=ap.parse_args()
    if a.phase=='selftest':
        selftest();return
    if a.output is None:a.output=str(Path(a.project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct14_p2588_forensic')
    require(1<=a.max_qa_panels<=1025,'Invalid QA panel limit')
    if a.phase=='inventory':
        paths=source_paths(a.project,a.coco_train)
        verify_sources(paths)
        rows,_,_=load_whitelist(paths['csv'],paths['spatial'],paths['fn'])
        print('[P2588 TRAIN COCO INVENTORY]',json.dumps(inventory_allowlist(paths['coco'],rows),sort_keys=True,indent=2))
        return
    if a.phase=='zip-inventory':
        paths=source_paths(a.project,a.coco_train)
        verify_sources(paths)
        rows,approved,_=load_whitelist(paths['csv'],paths['spatial'],paths['fn'])
        images,_=checked_authorized_coco(paths['coco'],rows,approved)
        with TrainImagesZip(a.images_zip) as src:
            # Only directory names and member paths are indexed. No image bytes opened.
            previews=[]
            for iid in sorted(images)[:3]:
                fn=src.clean_coco(images[iid]['file_name'])
                candidates=[name for name in src.members if (name==fn or name.endswith('/'+fn))
                            and 'train' in name.lower().split('/')[:-1]]
                previews.append({'image_id':iid,'COCO_file_name':fn,
                                 'train_candidate_count':len(candidates),'train_candidates':candidates[:3]})
            summary={'zip_path':str(src.path),'total_member_count':len(src.members),
                     'allowed_GT_image_ids':len(images),
                     'no_image_bytes_read':True,
                     'sample_COCO_to_ZIP_lookup':previews,
                     'note':'TEST/VAL member names indexed, but no TEST/VAL images opened'}
            print('[P2588 ZIP INVENTORY]',json.dumps(summary,indent=2))
        return
    if a.phase=='verify-build':
        dest=Path(a.output)/'P2588_identity_build'
        private_dest=private_output_destination(a)
        res=verify_published_build_pair(dest,private_dest,expected_runner_sha=current_runner_sha256(),allow_pinned_stage1_v11=True)
        print('[P2588 VERIFY-BUILD PASS]',json.dumps(res,indent=2))
        return
    if a.phase=='verify-adjudication':
        dest=Path(a.output)/'P2588_adjudication'
        private_adj_dest=private_adjudication_destination(a)
        res=verify_published_adjudication_pair(dest,private_adj_dest,expected_runner_sha=current_runner_sha256())
        print('[P2588 VERIFY-ADJUDICATION PASS]',json.dumps(res,indent=2))
        return
    if a.phase in ('review-preflight','adjudicate'):
        if a.phase=='adjudicate':
            require(bool(getattr(a,'authorize_stage2_adjudicate',False)),
                    'Stage 2 human adjudication on HOLD; pass --authorize-stage2-adjudicate only after explicit Project Director authorization')
        adjudicate(a)
        return
    if a.phase=='build':
        require(bool(getattr(a,'authorize_stage1_build',False)),
                'Stage 1 build is on HOLD; pass --authorize-stage1-build only after explicit Project Director authorization')
    build(a)


def selftest():
    a={'annotation_id':1,'video':18,'class':2,'frame':0,'source_w':1920.,'source_h':1080.,
       'x':20.,'y':30.,'w':8.,'h':6.,'cx':24.,'cy':33.,'area':48.,'is_tp':0}
    b={**a,'annotation_id':2,'frame':10,'x':21.,'cx':25.}
    require(proposal_cost(a,b) is not None,'Motion gate failed')
    require(proposal_cost(a,{**b,'video':7}) is None,'Cross-video link')
    require(proposal_cost(a,{**b,'class':3}) is None,'Cross-class link')
    require(proposal_cost(a,{**b,'x':600.,'cx':604.}) is None,'Huge displacement gate failed')
    ed,ac,am=provisional_edges([a,b]);require(len(ac)==1 and len(ed)==1,'Simple link failed')
    c,lk=provisional_components([a,b],ac);require(len(c)==1 and c[0]['verification_status']=='PROVISIONAL_UNVERIFIED',
                                            'Unverified cluster was treated as physical object')
    require(len(validate_human_decisions(c,[{**c[0], 'verification_status':'UNRESOLVED'}])[0])==0,
            'Unresolved must not count as verified')
    c2,_=provisional_components([a,{**b,'video':2,'class':0,'annotation_id':9,'frame':0}],[])
    ordered=order_qa_clusters(list(reversed(c2)))
    require([x['provisional_cluster_id'] for x in ordered]==['P0001','P0002'],
            'Outcome-independent cluster ordering failed')
    redacted=public_source_pins({'csv':{'path':r'D:\secret\train_spatial_units.csv','sha256':SPATIAL_CSV_SHA}})
    require('path' not in redacted['csv'] and redacted['csv']['sha256']==SPATIAL_CSV_SHA,
            'Public source pin path redaction failed')
    sanitized=sanitize_trace_snippet(r'transform = load("D:\MARIVISION\train_spatial_units.csv")')
    require('D:\\' not in sanitized and 'train_spatial_units.csv' not in sanitized,
            'Trace snippet path/filename redaction failed')
    require(len(current_runner_sha256())==64,'Runner SHA-256 computation failed')
    require(reviewer_attestation_mode('chatgpt_blind_qa_v1')=='AI_BLIND_QA_ATTESTED' and
            reviewer_attestation_mode('rater_human_1')=='HUMAN_ATTESTED',
            'Attestation mode classification failed')
    m_c1={'provisional_cluster_id':'P0001','video':2,'class':6,'class_name':'SHIP','n_observations':8,
          'first_frame':0,'last_frame':70,'annotation_ids':'1;2;3;4;5;6;7;8',
          'verification_status':'VERIFIED','physical_object_id':'OBJ_1','identity_evidence':'IMAGE_QA','verified_by':'chatgpt_blind_qa_v1'}
    m_c2={'provisional_cluster_id':'P0002','video':2,'class':6,'class_name':'SHIP','n_observations':8,
          'first_frame':80,'last_frame':150,'annotation_ids':'9;10;11;12;13;14;15;16',
          'verification_status':'VERIFIED','physical_object_id':'OBJ_2','identity_evidence':'IMAGE_QA','verified_by':'chatgpt_blind_qa_v1'}
    v_acc,_=validate_human_decisions([m_c1,m_c2],[m_c1,m_c2])
    v_m,comp_map,m_comps=validate_candidate_merges([m_c1,m_c2],v_acc,[{
        'video':2,'class':6,'class_name':'SHIP','earlier_cluster_id':'P0001','later_cluster_id':'P0002',
        'earlier_last_frame':70,'later_first_frame':80,'frame_gap':10,
        'visual_and_kinematic_rationale':'Same ship split across 2x scale gate','merge_confidence':'MEDIUM'
    }])
    require(len(v_m)==1 and len(m_comps)==1 and comp_map['P0001']==comp_map['P0002'],'Candidate merge consolidation failed')
    print('[PASS] CPU GT-only linking, complete outcome-independent QA panels, V12 Stage 2 ZIP-native blind adjudication, two-tree rater/outcome separation & merge protocol contracts verified')


if __name__=='__main__':main()
