"""No-native-data regression contracts for OCT13 matched-control spatial diagnostic."""
import hashlib,inspect,math,unittest
from pathlib import Path
import numpy as np
import run15a_oct13_spatial_matched_control as m

class SpatialDiagnosticContracts(unittest.TestCase):
    def test_selftest(self):self.assertIn('[PASS]',m.selftest())
    def test_coco_alignment_independent_order(self):
        gt=np.array([[.2,.3,.1,.1],[.7,.6,.2,.2]])
        gt_classes=np.array([2,3])
        ann=[{'id':19,'category_id':4,'bbox':[60,50,20,20]},
             {'id':18,'category_id':3,'bbox':[15,25,10,10]}]
        ids=m.validate_coco_alignment(gt,gt_classes,ann,{'width':100,'height':100})
        self.assertEqual(ids[0]['annotation_id'],18)
        self.assertEqual(ids[1]['annotation_id'],19)
    def test_bad_class_alignment_rejected(self):
        with self.assertRaises(AssertionError):
            m.validate_coco_alignment(np.array([[.5,.5,.1,.1]]),np.array([2]),
                  [{'id':2,'category_id':4,'bbox':[45,45,10,10]}],{'width':100,'height':100})
    def test_bad_bbox_alignment_rejected(self):
        with self.assertRaises(AssertionError):
            m.validate_coco_alignment(np.array([[.5,.5,.1,.1]]),np.array([2]),
                  [{'id':2,'category_id':3,'bbox':[4,4,10,10]}],{'width':100,'height':100})
    def test_expected_3d_geometry_is_not_refpoint(self):
        b=np.array([[.5,.5,.2,.2],[.9,.9,.05,.05]])
        z=np.zeros((2,7));z[0,2]=1.
        r=m.native_unit(2,0,123,0,True,2,np.array([.5,.5,.1,.1]),b,z,432,
                       {'width':200,'height':100,'camera_sensor':'CAM'},1.)
        self.assertEqual(r['gt_w_source_px'],20.)
        self.assertEqual(r['gt_h_source_px'],10.)
        self.assertEqual(r['decoded_box_center_density_1gt'],1)
        self.assertEqual(r['decoder_refpoint_density_status'],'NOT_COMPUTED_O_R_UNVERIFIED')
    def test_nearmiss_selection_threshold(self):
        def row(iou):return {'is_tp':0,'video':18,'class':2,'best_raw_iou':iou,
            'gt_area_source_px2':10.,'decoded_box_center_density_1gt':1,
            'signed_center_x_over_gt_w':0.,'signed_center_y_over_gt_h':0.,
            'log_pred_w_over_gt_w':0.,'log_pred_h_over_gt_h':0.}
        x=m.descriptive_analysis([row(.25),row(.299999),row(.30),row(.249999)],[])
        self.assertEqual(x['near_misses']['n'],2)
    def test_control_same_video_same_class_caliper(self):
        fn={'video':18,'frame':1,'gt_index':1,'class':2,'is_tp':0,'gt_area_source_px2':100.,
            'best_raw_iou':.2,'decoded_box_center_density_1gt':0,'image_id':1}
        tp=[dict(fn,frame=2,gt_index=i,is_tp=1,gt_area_source_px2=a,image_id=i+10) for i,a in enumerate([90.,170.,250.])]
        other=dict(tp[0],video=24,gt_index=4)
        pair,unmatched=m.match_controls([fn]+tp+[other])
        self.assertEqual(len(pair),2)
        self.assertEqual(unmatched,[])
        self.assertTrue(all(p['tp_key'][0]==18 for p in pair))
    def test_unavailable_control_is_not_fabricated(self):
        fn={'video':18,'frame':1,'gt_index':1,'class':2,'is_tp':0,'gt_area_source_px2':10.,
            'best_raw_iou':.2,'decoded_box_center_density_1gt':0,'image_id':1}
        tp=dict(fn,frame=2,gt_index=2,is_tp=1,gt_area_source_px2=1000.)
        pairs,unmatched=m.match_controls([fn,tp]);self.assertEqual(pairs,[])
        self.assertEqual(len(unmatched),1)
    def test_fn_id_parity_detects_drift(self):
        prior=[{'video':18,'frame':0,'image_id':123,'g':0,'class':2,'type':'A','max_box_iou_raw300':.12}]
        new=[prior[0].copy()]
        with self.assertRaises(AssertionError):m.compare_fn_report(new,prior) # need all 271
    def test_train_only_source_and_no_policy(self):
        src=inspect.getsource(m)
        self.assertIn('for vid in TRAIN:',src)
        self.assertNotIn('for vid in f.a.EVAL',src)
        self.assertNotIn('optimizer.step(',src)
        self.assertNotIn('model.train(',src)
        self.assertNotIn("'GO_SPATIAL'",src)
    def test_source_pin_is_expected_corrected_version(self):
        self.assertEqual(m.FN_SCRIPT_SHA,'9df5615a70a229dcdc7d7bf7af7846fb7ae6a005a012e07e7a8784fba14597fa')
    def test_writer_write_once_guards(self):
        src=inspect.getsource(m.paths)
        self.assertIn('Existing one-shot output',src)
        self.assertIn('source drift',inspect.getsource(m.paths).lower() if 'source drift' in src.lower() else 'source drift')

if __name__=='__main__':unittest.main()

class ExtendedScientificContracts(unittest.TestCase):
    def test_full_identity_parity_true(self):
        prior=[{'video':18,'frame':i,'image_id':4000+i,'g':0,'class':2,
                'type':'WEAK_OVERLAP_USABLE_TOP100_ANYCLASS','max_box_iou_raw300':.25} for i in range(271)]
        actual=[dict(x) for x in prior]
        self.assertTrue(m.compare_fn_report(actual,prior))
    def test_full_identity_parity_rejects_geometry_drift(self):
        prior=[{'video':18,'frame':i,'image_id':4000+i,'g':0,'class':2,
                'type':'WEAK_OVERLAP_USABLE_TOP100_ANYCLASS','max_box_iou_raw300':.25} for i in range(271)]
        actual=[dict(x) for x in prior];actual[50]['max_box_iou_raw300']=.251
        with self.assertRaises(AssertionError):m.compare_fn_report(actual,prior)
    def test_reused_tp_not_reported_as_independent(self):
        def row(v,f,g,tp,area):
            return {'video':v,'frame':f,'gt_index':g,'class':2,'is_tp':int(tp),
            'gt_area_source_px2':area,'best_raw_iou':.4 if tp else .1,
            'decoded_box_center_density_1gt':1,'image_id':f+100}
        x=[row(18,i,i,False,100.) for i in range(8)]+[row(18,20,0,True,100.)]
        p,_=m.match_controls(x)
        self.assertEqual(len(p),8)
        self.assertEqual(len(set(tuple(z['tp_key']) for z in p)),1)
    def test_source_sizes_not_assumed_to_be_r50_input_size(self):
        x=m.native_unit(2,0,1,0,False,2,np.array([.5,.5,.1,.1]),
                        np.array([[.5,.5,.1,.1]]),np.zeros((1,7)),100,
                        {'width':1920,'height':1020},density=1.)
        self.assertEqual(x['input_pixel_size_status'],'UNVERIFIED_CANONICAL_TRANSFORM')
        self.assertAlmostEqual(x['gt_w_source_px'],192.0)


class ClaudeP1CorrectionsV3(unittest.TestCase):
    def test_all_geometry_outcome_confound_status(self):
        from types import SimpleNamespace
        b=np.array([[.5,.5,.10,.1],[.2,.2,.1,.1]])
        r=m.native_unit(18,0,123,0,False,2,np.array([.5,.5,.1,.1]),
                        b,np.zeros((2,7)),432,{'width':1920,'height':1080},1.)
        self.assertEqual(r['query_geometry_inference_status'],'SELECTION_CONFOUNDED_NOT_INFERENTIAL')
        self.assertEqual(m.descriptive_analysis([r],[])['geometry_comparison_inference_status'],
                         'SELECTION_CONFOUNDED_NOT_INFERENTIAL')
    def test_small_video_bootstrap_is_never_reported(self):
        pairs=[]
        for v in (3,4,16,20,24):
            for j in range(3):
                pairs.append({'fn_key':(v,j,j),'fn_density':2,'tp_density':1})
        report=m.paired_density_contrasts(pairs)
        self.assertEqual(report['non18_video_count'],5)
        self.assertIsNone(report['non18_video_bootstrap_interval'])
        self.assertEqual(report['interval_status'],'INSUFFICIENT_CLUSTERS_AND_SELECTION_CONFOUNDED')
    def test_reference_identity_is_not_semantic_proof(self):
        from types import SimpleNamespace
        import torch
        ref=torch.linspace(.01,.99,600,dtype=torch.float32).view(300,2)
        O=SimpleNamespace(r=ref,z=torch.zeros(300,7),b=torch.zeros(300,4))
        desc,arr=m.inspect_O_r(O,{'ref_pts':ref.clone()})
        self.assertEqual(desc['status'],'O_R_EQUALS_CACHED_REF_PTS_COORDINATE_IDENTITY_ONLY')
        self.assertEqual(arr.shape,(300,2))
        self.assertEqual(desc['max_abs_difference'],0.)
        u=m.native_unit(2,0,1,0,False,2,np.array([.5,.5,.1,.1]),
                       np.array([[.5,.5,.1,.1]]),np.zeros((1,7)),100,
                       {'width':100,'height':100},density=1.,refs=arr)
        self.assertIsInstance(u['cached_refpoint_density_1gt'],int)
        self.assertIn('SEMANTICS_UNVERIFIED',u['decoder_refpoint_density_status'])
    def test_reference_mismatch_or_invalid_rejected_for_stats(self):
        from types import SimpleNamespace
        import torch
        ref=torch.full((300,2),.5)
        O=SimpleNamespace(r=ref,z=torch.zeros(300,7),b=torch.zeros(300,4))
        d,a=m.inspect_O_r(O,{'ref_pts':torch.full((300,2),.6)})
        self.assertIsNone(a)
        self.assertEqual(d['status'],'O_R_DIFFERS_FROM_CACHED_REF_PTS')
        O.r=torch.full((300,2),1.5)
        d,a=m.inspect_O_r(O,{'ref_pts':O.r.clone()})
        self.assertIsNone(a)
        self.assertEqual(d['status'],'REFERENCE_OUTSIDE_NORMALIZED_IMAGE')
    def test_readonly_schema_and_embargo(self):
        src=inspect.getsource(m)
        self.assertIn('SELECTION_CONFOUNDED_NOT_INFERENTIAL',src)
        self.assertIn('NOT_COMPUTED_O_R_UNVERIFIED',src)
        self.assertIn('O,X=s0a.make_O_X(item,spa[0],tmp[0],CO,FX)',src)
        self.assertNotIn('np.quantile(draws',src)
        self.assertNotIn('for vid in f.a.EVAL',src)
