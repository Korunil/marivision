"""Scientific regression/negative tests for P-2588. Runs CPU-only with stdlib + Pillow optional."""
import csv,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_p2588_object_forensics as m


def gt(aid,frame=0,v=18,c=2,x=10,y=20,w=8,h=6,tp=0):
    return {'annotation_id':aid,'frame':frame,'video':v,'class':c,
            'x':float(x),'y':float(y),'w':float(w),'h':float(h),
            'cx':x+w/2,'cy':y+h/2,'area':w*h,'source_w':1920.,'source_h':1080.,
            'is_tp':tp,'image_id':100+frame}


class GTLinkingContracts(unittest.TestCase):
    def test_no_cross_video_class_or_discontinuous_time(self):
        a=gt(1);b=gt(2,10)
        self.assertIsNone(m.proposal_cost(a,{**b,'video':7}))
        self.assertIsNone(m.proposal_cost(a,{**b,'class':3}))
        self.assertIsNone(m.proposal_cost(a,{**b,'frame':20}))
    def test_size_jump_rejected(self):
        a=gt(1);b=gt(2,10,w=40,h=40)
        self.assertIsNone(m.proposal_cost(a,b))
    def test_motion_jump_rejected(self):
        self.assertIsNone(m.proposal_cost(gt(1),gt(2,10,x=950)))
    def test_src_dims_change_fails_closed(self):
        with self.assertRaisesRegex(AssertionError,'dimensions changed'):
            m.proposal_cost(gt(1),{**gt(2,10),'source_w':1280.})
    def test_mutual_unique_pair_is_provisional(self):
        e,a,amb=m.provisional_edges([gt(1),gt(2,10,x=11)])
        self.assertEqual((len(e),len(a),amb),(1,1,0))
        self.assertEqual(e[0]['linkage_status'],'PROPOSED_UNVERIFIED')
        cl,lookup=m.provisional_components([gt(1),gt(2,10,x=11)],a)
        self.assertEqual(len(cl),1)
        self.assertEqual(cl[0]['physical_object_id'],'')
    def test_ambiguous_equal_cost_is_not_joined(self):
        units=[gt(1),gt(2,10,x=11),gt(3,10,x=11)]
        all_edges,edges,namb=m.provisional_edges(units)
        self.assertEqual(len(edges),0)
        self.assertEqual(len(all_edges),2)
        self.assertEqual(namb,2)
        self.assertEqual(len(m.provisional_components(units,edges)[0]),3)
    def test_no_same_frame_duplicates_in_cluster(self):
        c,_=m.provisional_components([gt(1),gt(2,10)],[])
        self.assertEqual(len(c),2)
    def test_known_track_id_not_annotation_id(self):
        self.assertEqual(m.coco_track_key({'id':500}),())
        self.assertEqual(m.coco_track_key({'id':500,'track_id':'T2'}),(('track_id','T2'),))
    def test_nn_distance_framewise_not_predictionwise(self):
        aa=[gt(1),gt(2,x=50),gt(3,frame=10)]
        by={100:aa[:2],110:[aa[2]]}
        result=m.gt_neighbors(aa,by)
        self.assertAlmostEqual(result[0]['nearest_anyclass_gt_center_source_px'],40.)
        self.assertIsNone(result[2]['nearest_anyclass_gt_center_source_px'])
    def test_bounds_not_unique_object_claim(self):
        rows=[gt(1),gt(2),gt(3,10),gt(4,10)]
        z=m.report_cluster_bounds(rows)[0]
        self.assertEqual(z['lower_bound_independent_objects_gt_only'],2)
        self.assertEqual(z['upper_bound_independent_objects_gt_only'],4)


class AuditPolicyContracts(unittest.TestCase):
    def test_bad_bbox_rejected(self):
        row={'annotation_id':1,'image_id':10,'video':18,'frame':0,'class':2,'gt_index':0,
             'is_tp':0,'gt_area_source_px2':'25','best_raw_query':'10',
             'signed_center_x_over_gt_w':'0','signed_center_y_over_gt_h':'0',
             'log_pred_w_over_gt_w':'0','log_pred_h_over_gt_h':'0'}
        image={10:{'id':10,'video_id':18,'frame_id':0,'width':100,'height':100}}
        ann={1:{'id':1,'image_id':10,'category_id':3,'bbox':[10,10,6,6]}}
        with self.assertRaisesRegex(AssertionError,'area mismatch'):
            m.materialize_gt([row],image,ann)
    def test_bad_class_rejected(self):
        row={'annotation_id':1,'image_id':10,'video':18,'frame':0,'class':2,'gt_index':0,
             'is_tp':0,'gt_area_source_px2':'36','best_raw_query':'10',
             'signed_center_x_over_gt_w':'0','signed_center_y_over_gt_h':'0',
             'log_pred_w_over_gt_w':'0','log_pred_h_over_gt_h':'0'}
        image={10:{'id':10,'video_id':18,'frame_id':0,'width':100,'height':100}}
        ann={1:{'id':1,'image_id':10,'category_id':4,'bbox':[10,10,6,6]}}
        with self.assertRaisesRegex(AssertionError,'category/image mismatch'):
            m.materialize_gt([row],image,ann)
    def test_forbidden_video_is_not_train(self):
        self.assertFalse(bool(set(m.TRAIN)&m.FORBIDDEN))
        self.assertFalse(38 in m.TRAIN)
        self.assertEqual(len(m.TRAIN),17)
    def test_stale_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'existing';p.mkdir()
            with self.assertRaisesRegex(AssertionError,'already exists'):
                m.tmp_output(p)
    def test_absolute_and_parent_image_paths_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td,'okay.png').write_bytes(b'x')
            with self.assertRaisesRegex(AssertionError,'absolute'):
                m.image_path(td,'C:/any/image.png')
            with self.assertRaisesRegex(AssertionError,'escapes'):
                m.image_path(td,'../notallow.png')
    def test_gt_oracle_is_not_native_prediction(self):
        z=gt(1)
        z.update({'signed_center_x_over_gt_w':0,'signed_center_y_over_gt_h':0,
                  'log_pred_w_over_gt_w':math.log(2), 'log_pred_h_over_gt_h':0})
        box=m.oracle_box_from_unit(z)
        self.assertAlmostEqual(box['w'],16.)
        self.assertAlmostEqual(box['x'],6.)
        self.assertIn('ORACLE',m.oracle_box_from_unit.__doc__)
    def test_static_trace_does_not_certify_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            m.write_trace(Path(td),td)
            text=(Path(td)/'source_decoder_trace.md').read_text()
            self.assertIn('UNVERIFIED',text)
            self.assertIn('No detector forward',text)


class SourceImageAllowlistContracts(unittest.TestCase):
    def test_272_authorized_images_even_when_many_have_no_gt(self):
        with tempfile.TemporaryDirectory() as td:
            images=[{'id':i+100000+v*1000,'video_id':v,'frame_id':i*10,
                     'height':1080,'width':1920} for v in m.TRAIN for i in range(16)]
            x=images[0]
            data={'categories':[{'id':i+1,'name':name} for i,name in enumerate(m.CLASS_NAMES)],
                  'images':images,
                  'annotations':[{'id':1,'image_id':x['id'],'category_id':3,'bbox':[1,1,3,3]}]}
            p=Path(td)/'train.json';p.write_text(json.dumps(data))
            got,ann=m.checked_authorized_coco(p,[{'annotation_id':'1','image_id':str(x['id']),'video':str(x['video_id']),'frame':str(x['frame_id']),'class':'2'}],{x['id']:[{}]})
            self.assertEqual(len(got),1)
            self.assertEqual(len(ann),1)
    def test_extra_gt_in_allowlisted_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            images=[{'id':i+100000+v*1000,'video_id':v,'frame_id':i*10,
                     'height':1080,'width':1920} for v in m.TRAIN for i in range(16)]
            x=images[0]
            data={'categories':[{'id':i+1,'name':name} for i,name in enumerate(m.CLASS_NAMES)],
                  'images':images,
                  'annotations':[{'id':1,'image_id':x['id'],'category_id':3,'bbox':[1,1,3,3]},
                                 {'id':2,'image_id':x['id'],'category_id':3,'bbox':[5,5,4,4]}]}
            p=Path(td)/'train.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError,'extra_annotations_on_exact_gt_bearing_images_count'):
                m.checked_authorized_coco(p,[{'annotation_id':'1','image_id':str(x['id']),'video':str(x['video_id']),'frame':str(x['frame_id']),'class':'2'}],{x['id']:[{}]})


class HumanReviewContracts(unittest.TestCase):
    def setUp(self):
        units=[gt(1),gt(2,10)]
        edges=m.provisional_edges(units)[1]
        self.c=m.provisional_components(units,edges)[0][0]
    def test_unreviewed_not_accepted(self):
        a,_=m.validate_human_decisions([self.c],[{**self.c,'verification_status':'UNRESOLVED'}])
        self.assertEqual(a,[])
    def test_verified_requires_human_identity_and_evidence(self):
        with self.assertRaisesRegex(AssertionError,'VERIFIED needs'):
            m.validate_human_decisions([self.c],[{**self.c,'verification_status':'VERIFIED'}])
    def test_verified_has_explicit_attestation(self):
        a,_=m.validate_human_decisions([self.c],[{
            **self.c,'verification_status':'VERIFIED','physical_object_id':'REAL_1',
            'verified_by':'independent_reviewer','identity_evidence':'IMAGE_QA'}])
        self.assertEqual(a[0]['verification_status'],'HUMAN_ATTESTED')
    def test_review_manifest_tampering_fails(self):
        with self.assertRaisesRegex(AssertionError,'Tampered'):
            m.validate_human_decisions([self.c],[{**self.c,'class':0,'verification_status':'UNREVIEWED'}])
    def test_review_missing_cluster_fails(self):
        with self.assertRaisesRegex(AssertionError,'every provisional'):
            m.validate_human_decisions([self.c],[])
    def test_object_ids_not_merged_without_adjudication(self):
        one={**self.c};two={**self.c,'provisional_cluster_id':'P0999',
                            'annotation_ids':'999','n_observations':1}
        r=[{**a,'verification_status':'VERIFIED','physical_object_id':'SAME',
            'verified_by':'r','identity_evidence':'IMAGE_QA'} for a in (one,two)]
        with self.assertRaisesRegex(AssertionError,'silently merge'):
            m.validate_human_decisions([one,two],r)
    def test_stratified_seed_reproducible_unique(self):
        clusters=[{**self.c,'physical_object_id':f'X{i}','provisional_cluster_id':f'P{i:04d}',
                   'n_fn':i%3,'video':18 if i%2 else 24,'class':2 if i%2 else 3}
                  for i in range(54)]
        a=m.select_blind_cases(clusters);b=m.select_blind_cases(clusters)
        self.assertEqual([x['physical_object_id'] for x in a],
                         [x['physical_object_id'] for x in b])
        self.assertEqual(len(a),40)
        self.assertEqual(len(set(x['physical_object_id'] for x in a)),40)

if __name__=='__main__':unittest.main(verbosity=2)

class OptionalBlindImageTests(unittest.TestCase):
    def test_blind_panels_are_rendered_without_native_prediction_claim(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow unavailable')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            p=root/'vid18.png';Image.new('RGB',(200,120),'white').save(p)
            z=gt(88,w=8,h=6)
            z.update({'image_id':501,'signed_center_x_over_gt_w':0.,
                      'signed_center_y_over_gt_h':0.,
                      'log_pred_w_over_gt_w':.2,'log_pred_h_over_gt_h':.3})
            img={501:{'file_name':'vid18.png'}}
            case={'physical_object_id':'V18_MARKER_1','annotation_ids':'88'}
            key,ratings=m.render_blind_panels(root,[case],[z],img,root)
            self.assertEqual(len(key),1)
            self.assertEqual(len(ratings),4)
            self.assertEqual(key[0]['native_score_ranked_prediction'],'NOT_AVAILABLE_NO_NEW_FORWARD')
            self.assertEqual(set((key[0]['panel_A_semantics'],key[0]['panel_B_semantics'])),
                             {'GT','ORACLE'})
            self.assertTrue((root/'stage1_blind'/'CASE_001.png').is_file())
            self.assertTrue((root/'stage2_blind'/'CASE_001_A.png').is_file())

class CorrectedSourceCoverageContracts(unittest.TestCase):
    def make_case(self, root, trim_empty=False, duplicate_slot=False):
        images=[{'id':100000+v*1000+i,'video_id':v,'frame_id':10*i,'height':1080,'width':1920}
                for v in m.TRAIN for i in range(16)]
        # Mirrors the 248 GT-bearing images: no GT in video31, and none in video8 frames 0..70.
        annotated=[im for im in images if im['video_id']!=31 and
                   (im['video_id']!=8 or im['frame_id']>=80)]
        anns=[{'id':n,'image_id':im['id'],'category_id':3,'bbox':[1,2,3,4]}
              for n,im in enumerate(annotated,1)]
        if trim_empty:
            images=annotated[:]
        if duplicate_slot:
            images.append(dict(images[0]))
        data={'categories':[{'id':i+1,'name':name} for i,name in enumerate(m.CLASS_NAMES)],
              'images':images,'annotations':anns}
        p=root/'train.json';p.write_text(json.dumps(data),encoding='utf-8')
        rows=[{'annotation_id':str(a['id']),'image_id':str(a['image_id']),
               'video':str(im['video_id']),'frame':str(im['frame_id']),'class':'2'}
               for a,im in zip(anns,annotated)]
        return p, rows,{int(a['image_id']):[] for a in anns}
    def test_248_coco_records_suffice_for_all_248_gt_bearing_images(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=self.make_case(Path(td),trim_empty=True)
            images,anns=m.checked_authorized_coco(p,rows,allowed)
            self.assertEqual((len(images),len(anns)),(248,248))
            d=m.inventory_allowlist(p,rows)
            self.assertEqual(d['frozen_cached_sampled_frame_count'],272)
            self.assertEqual(d['coco_images_with_exact_cached_ids'],248)
            self.assertEqual(d['zero_gt_cached_frames_NOT_resolved_to_coco_image_records'],24)
            self.assertEqual(d['missing_exact_cached_image_count'],0)
            self.assertEqual(d['duplicate_source_image_ids_count'],0)
    def test_full_272_coco_records_also_pass(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=self.make_case(Path(td))
            images,anns=m.checked_authorized_coco(p,rows,allowed)
            self.assertEqual((len(images),len(anns)),(248,248))
            self.assertEqual(m.inventory_allowlist(p,rows)['missing_exact_cached_image_count'],0)
    def test_duplicate_coco_video_frame_slot_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=self.make_case(Path(td),duplicate_slot=True)
            with self.assertRaisesRegex(AssertionError,'duplicate_source_image_ids_count'):
                m.checked_authorized_coco(p,rows,allowed)
    def test_missing_gt_bearing_coco_image_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=self.make_case(Path(td),trim_empty=True)
            raw=json.loads(p.read_text())
            raw['images']=raw['images'][1:]
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(AssertionError,'missing_exact_cached_image_count'):
                m.checked_authorized_coco(p,rows,allowed)
    def test_coco_image_frame_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=self.make_case(Path(td),trim_empty=True)
            rows[0]['frame']='10'
            with self.assertRaisesRegex(AssertionError,'COCO frame_id minus cached index not constant'):
                m.checked_authorized_coco(p,rows,allowed)



class SourceFrameOffsetV3(unittest.TestCase):
    def test_exact_cached_ids_permit_nonzero_video_dependent_source_offset(self):
        with tempfile.TemporaryDirectory() as td:
            helper=CorrectedSourceCoverageContracts()
            p,rows,allowed=helper.make_case(Path(td))
            raw=json.loads(p.read_text())
            for im in raw['images']:
                im['frame_id']+=7+(int(im['video_id'])%11)
            p.write_text(json.dumps(raw))
            got,anns=m.checked_authorized_coco(p,rows,allowed)
            self.assertEqual(len(got),248)
            self.assertEqual(len(anns),248)
            diag=m.inventory_allowlist(p,rows)
            self.assertEqual(diag['missing_exact_cached_image_count'],0)
            self.assertEqual(diag['exact_annotation_ids_in_full_source'],248)
            self.assertNotEqual(diag['per_video']['18']['source_frame_offsets'],[0])
            self.assertEqual(got[int(rows[0]['image_id'])]['_p2588_cached_frame_idx'],0)
    def test_nonconstant_source_offset_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=CorrectedSourceCoverageContracts().make_case(Path(td))
            raw=json.loads(p.read_text());raw['images'][1]['frame_id']+=1
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(AssertionError,'not constant'):
                m.checked_authorized_coco(p,rows,allowed)
    def test_wrong_source_video_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=CorrectedSourceCoverageContracts().make_case(Path(td))
            raw=json.loads(p.read_text());raw['images'][0]['video_id']=99
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(AssertionError,'Source COCO video_id differs'):
                m.checked_authorized_coco(p,rows,allowed)
    def test_exact_source_annotation_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=CorrectedSourceCoverageContracts().make_case(Path(td))
            raw=json.loads(p.read_text());raw['annotations'][0]['image_id']=77777
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(AssertionError,'wrong_image_or_category_annotation_ids_count'):
                m.checked_authorized_coco(p,rows,allowed)
    def test_source_unlisted_extra_annotation_on_valid_gt_image_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,rows,allowed=CorrectedSourceCoverageContracts().make_case(Path(td))
            raw=json.loads(p.read_text());raw['annotations'].append(
                {'id':9999,'image_id':int(rows[0]['image_id']),'category_id':3,'bbox':[1,2,3,4]})
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(AssertionError,'extra_annotations_on_exact_gt_bearing_images_count'):
                m.checked_authorized_coco(p,rows,allowed)

class ClaudeP1BuildPrivacyV4(unittest.TestCase):
    def test_neighbor_table_has_no_outcome_flag(self):
        a=gt(4,tp=1); b=gt(5,frame=10,tp=0)
        self.assertNotIn('is_tp_descriptive',m.gt_neighbors([a,b],{100:[a],110:[b]})[0])
        self.assertFalse(any('tp' in k.lower() or 'fn' in k.lower()
                             for k in m.gt_neighbors([a,b],{100:[a],110:[b]})[0]))

    def test_image_root_is_mandatory(self):
        with self.assertRaisesRegex(AssertionError,'--images-root required'):
            m.verify_train_images(None, {}, [])

    def test_sealed_root_is_mandatory_and_separate(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            shared=Path(td)/'share'
            with self.assertRaisesRegex(AssertionError,'--sealed-root required'):
                m.private_output_destination(SimpleNamespace(output=str(shared),sealed_root=None))
            for secret in (shared,shared/'child',Path(td)):
                with self.assertRaisesRegex(AssertionError,'separate non-nested'):
                    m.private_output_destination(SimpleNamespace(output=str(shared),sealed_root=str(secret)))
            dest=m.private_output_destination(SimpleNamespace(output=str(shared),sealed_root=str(Path(td)/'secret')))
            self.assertTrue(dest.is_relative_to(Path(td)/'secret'))
            self.assertFalse(dest.is_relative_to(shared))

    def test_image_header_dimensions_and_full_allowlist(self):
        try:
            from PIL import Image
        except ImportError:self.skipTest('Pillow unavailable')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            metas={}
            for i in range(248):
                name=f'auth_{i}.png'; Image.new('RGB',(24,20),'white').save(root/name)
                metas[300+i]={'file_name':name,'width':24,'height':20}
            result=m.verify_train_images(root,metas,metas)
            self.assertEqual(result['n_verified_GT_bearing_source_image_headers'],248)
            metas[300]['width']=25
            with self.assertRaisesRegex(AssertionError,'dimensions differ'):
                m.verify_train_images(root,metas,metas)
            metas[300]['width']=24
            metas[301]['file_name']=metas[300]['file_name']
            with self.assertRaisesRegex(AssertionError,'same file'):
                m.verify_train_images(root,metas,metas)
            metas[301]['file_name']='auth_1.png'
            (root/'auth_7.png').unlink()
            with self.assertRaisesRegex(AssertionError,'missing'):
                m.verify_train_images(root,metas,metas)

    def test_stage2_is_disabled_in_main(self):
        import unittest.mock as mock
        with mock.patch.object(sys,'argv',['runner','--phase','adjudicate']):
            with self.assertRaisesRegex(AssertionError,'Stage 2 human adjudication on HOLD'):
                m.main()

    def test_qa_subpixel_origin_matches_integer_crop(self):
        # A GT offset of 0.8px must not be lost by truncating the crop origin
        # while computing overlay positions against the float ROI.
        import inspect
        src=inspect.getsource(m.identity_qa_panels)
        self.assertIn("x0=max(0,int(z['cx']-r))",src)
        self.assertIn("x1=min(im.width,int(z['cx']+r))",src)
        self.assertNotIn("z['cx']+r)-x0",src)

    def test_categories_logged_from_raw_file(self):
        import inspect
        src=inspect.getsource(m.build)
        self.assertIn("'raw_source_categories':raw_coco['categories']",src)

    def test_no_sealed_outcomes_written_to_review_dir(self):
        import inspect
        src=inspect.getsource(m.build)
        self.assertNotIn("write_csv(tmp/'SEALED_cluster_outcomes",src)
        self.assertIn("write_csv(private_tmp/'SEALED_cluster_outcomes",src)

class ZipNativeV5Contracts(unittest.TestCase):
    @staticmethod
    def fixture(root,n=248,include_test=True):
        from PIL import Image
        import io,zipfile
        zpath=root/'SeaClips.zip'
        metas={}
        with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for k in range(n):
                iid=1000+k
                fn=f'video_18/frame_{k:04d}.png'
                member='SeaClips/train/'+fn
                b=io.BytesIO();Image.new('RGB',(32,24),color=(k%255,3,4)).save(b,format='PNG')
                archive.writestr(member,b.getvalue())
                if include_test and k==0:
                    archive.writestr('SeaClips/test/'+fn,b'NEVER OPEN ME')
                metas[iid]={'id':iid,'file_name':fn,'width':32,'height':24}
        return zpath,metas

    def test_zip_prefix_resolves_coalesced_dataset_without_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            z,metas=self.fixture(Path(td))
            with m.TrainImagesZip(z) as reader:
                receipt=m.verify_train_images_zip(reader,metas,metas)
                self.assertEqual(receipt['n_verified_GT_bearing_source_image_headers'],248)
                self.assertEqual(len(reader.image_members),248)
                self.assertTrue(all('/train/' in x for x in reader.image_members.values()))
                self.assertEqual(reader.rgb(metas[1000]).size,(32,24))
                self.assertEqual(len(reader.sha_by_id),248)
                self.assertTrue(receipt['authorized_image_content_receipt_sha256'])

    def test_zip_archive_test_only_member_is_not_accepted(self):
        from PIL import Image
        import io,zipfile
        with tempfile.TemporaryDirectory() as td:
            z,metas=self.fixture(Path(td),include_test=False)
            with zipfile.ZipFile(z,'a') as a:
                k='SeaClips/train/'+metas[1000]['file_name']
                # The test split should NOT ever provide a missing TRAIN source.
                a.writestr('SeaClips/val/video_18/val_only.png',b'invalid image')
            metas[1000]['file_name']='video_18/val_only.png'
            with m.TrainImagesZip(z) as reader:
                with self.assertRaisesRegex(AssertionError,'resolve uniquely'):
                    reader.resolve(metas,metas)

    def test_duplicate_train_candidates_fail_closed(self):
        import zipfile
        with tempfile.TemporaryDirectory() as td:
            z,metas=self.fixture(Path(td))
            with zipfile.ZipFile(z,'a') as a:
                a.writestr('ALTERNATE/train/'+metas[1000]['file_name'],b'other')
            with m.TrainImagesZip(z) as reader:
                with self.assertRaisesRegex(AssertionError,'resolve uniquely'):
                    reader.resolve(metas,metas)

    def test_wrong_dimensions_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            z,metas=self.fixture(Path(td))
            metas[1000]['width']=33
            with m.TrainImagesZip(z) as reader:
                with self.assertRaisesRegex(AssertionError,'dimensions differ'):
                    m.verify_train_images_zip(reader,metas,metas)

    def test_zip_image_pixel_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            z,metas=self.fixture(Path(td))
            with m.TrainImagesZip(z) as reader:
                m.verify_train_images_zip(reader,metas,metas)
                reader.sha_by_id[1000]='0'*64
                with self.assertRaisesRegex(AssertionError,'content changed'):
                    reader.rgb(metas[1000])

    def test_relative_traversal_and_absolute_names_rejected(self):
        for bad in ('../train/foo.png','video_18/../test/a.png','C:/train/a.png','/train/b.png'):
            with self.assertRaises(AssertionError):m.TrainImagesZip.clean_coco(bad)

    def test_unallowlisted_id_refuses_read(self):
        with tempfile.TemporaryDirectory() as td:
            z,metas=self.fixture(Path(td))
            with m.TrainImagesZip(z) as reader:
                reader.resolve(metas,metas)
                with self.assertRaisesRegex(AssertionError,'Unallowlisted'):
                    reader.read_bytes(999999)

    def test_image_zip_cli_defaults_to_original_user_archive(self):
        import inspect
        source=inspect.getsource(m.main)
        self.assertIn('D:\\MARIVISION\\SeaClips.zip',source)
        self.assertIn("a.phase=='zip-inventory'",source)
        self.assertIn('Stage 2 human adjudication on HOLD',source)


class V6CompleteOutcomeIndependentQAPanelsContracts(unittest.TestCase):
    @staticmethod
    def make_clusters_and_units(tp_flags=(0,1,0,1,1,0),extra_meta=False):
        specs=[
            (101, 2, 1, 0, 20, 20, 10, 8, tp_flags[0], 1001),
            (102, 4, 2, 0, 30, 30, 12, 10, tp_flags[1], 1002),
            (103, 4, 2, 10, 31, 30, 12, 10, tp_flags[2], 1003),
            (104, 7, 5, 0, 40, 40, 14, 12, tp_flags[3], 1004),
            (105, 18, 0, 0, 50, 50, 8, 6, tp_flags[4], 1005),
            (106, 18, 2, 0, 60, 60, 8, 6, tp_flags[5], 1006),
        ]
        units=[]
        for idx,(aid,vid,cl,fr,x,y,w,h,tp,iid) in enumerate(specs):
            u=gt(aid,frame=fr,v=vid,c=cl,x=x,y=y,w=w,h=h,tp=tp)
            u['image_id']=iid
            u['best_raw_query']=str(299-idx if extra_meta else idx)
            u['signed_center_x_over_gt_w']=0.5 if extra_meta else -0.1
            u['signed_center_y_over_gt_h']=-0.4 if extra_meta else 0.2
            u['log_pred_w_over_gt_w']=0.3 if extra_meta else 0.0
            u['log_pred_h_over_gt_h']=-0.2 if extra_meta else 0.0
            if extra_meta:
                u['oracle_iou']=0.01 if tp==0 else 0.85
                u['fn_severity']='HIGH' if tp==0 else 'NONE'
            units.append(u)
        _,accepted,_=m.provisional_edges(units)
        clusters,_=m.provisional_components(units,accepted)
        if extra_meta:
            for c in clusters:
                c['n_fn']=99 if c['video']==18 else 0
                c['n_tp']=0 if c['video']==18 else 99
                c['outcome_priority_rank']=0 if c['video']==18 else 10
        return clusters,units

    @staticmethod
    def write_test_images(root,units):
        from PIL import Image
        images={}
        for u in units:
            iid=u['image_id']
            if iid not in images:
                fn=f'img_{iid}.png'
                Image.new('RGB',(160,120),color=(iid%200,40,80)).save(root/fn)
                images[iid]={'id':iid,'file_name':fn,'width':160,'height':120}
        return images

    def test_default_max_qa_panels_covers_all_possible_clusters(self):
        import inspect
        self.assertGreaterEqual(m.MAX_QA_PANELS_DEFAULT,240)
        self.assertEqual(m.MAX_QA_PANELS_DEFAULT,1025)
        sig=inspect.signature(m.identity_qa_panels)
        self.assertEqual(sig.parameters['max_panels'].default,1025)

    def test_insufficient_max_qa_panels_fails_closed_before_writing_panels(self):
        clusters,units=self.make_clusters_and_units()
        self.assertEqual(len(clusters),5)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=self.write_test_images(root,units)
            out=root/'out'
            out.mkdir()
            with self.assertRaisesRegex(AssertionError,'max_qa_panels .* must be >= cluster_count'):
                m.identity_qa_panels(out,clusters,units,images,root,max_panels=4)
            self.assertFalse((out/'identity_qa_panels').exists())

    def test_renders_exactly_one_panel_per_provisional_cluster(self):
        clusters,units=self.make_clusters_and_units()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=self.write_test_images(root,units)
            out=root/'out';out.mkdir()
            rendered=m.identity_qa_panels(out,clusters,units,images,root,max_panels=len(clusters))
            self.assertEqual(rendered,len(clusters))
            panel_files=sorted(p.name for p in (out/'identity_qa_panels').glob('*.png'))
            expected=[f"{c['provisional_cluster_id']}.png" for c in clusters]
            self.assertEqual(panel_files,expected)

    def test_duplicate_or_empty_clusters_fail_closed(self):
        clusters,units=self.make_clusters_and_units()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=self.write_test_images(root,units)
            out=root/'out';out.mkdir()
            with self.assertRaisesRegex(AssertionError,'must not be empty'):
                m.identity_qa_panels(out,[],units,images,root,max_panels=10)
            dup=[clusters[0],{**clusters[1],'provisional_cluster_id':clusters[0]['provisional_cluster_id']}]
            with self.assertRaisesRegex(AssertionError,'unique non-empty provisional_cluster_id'):
                m.identity_qa_panels(out,dup,units,images,root,max_panels=10)

    def test_preexisting_panel_file_in_destination_fails_closed(self):
        clusters,units=self.make_clusters_and_units()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=self.write_test_images(root,units)
            out=root/'out';(out/'identity_qa_panels').mkdir(parents=True)
            (out/'identity_qa_panels'/'STALE.png').write_bytes(b'123')
            with self.assertRaisesRegex(AssertionError,'already contains PNG files'):
                m.identity_qa_panels(out,clusters,units,images,root,max_panels=10)

    def test_panel_ordering_is_outcome_independent_and_invariant_to_tp_fn_and_metadata(self):
        import hashlib
        clusters_a,units_a=self.make_clusters_and_units(tp_flags=(0,0,0,0,0,0),extra_meta=False)
        clusters_b,units_b=self.make_clusters_and_units(tp_flags=(1,1,1,1,1,1),extra_meta=True)
        clusters_c,units_c=self.make_clusters_and_units(tp_flags=(1,0,1,0,0,1),extra_meta=True)
        permuted_b=list(reversed(clusters_b))
        permuted_c=[clusters_c[4],clusters_c[1],clusters_c[3],clusters_c[0],clusters_c[2]]
        order_a=[c['provisional_cluster_id'] for c in m.order_qa_clusters(clusters_a)]
        order_b=[c['provisional_cluster_id'] for c in m.order_qa_clusters(permuted_b)]
        order_c=[c['provisional_cluster_id'] for c in m.order_qa_clusters(permuted_c)]
        self.assertEqual(order_a,['P0001','P0002','P0003','P0004','P0005'])
        self.assertEqual(order_a,order_b)
        self.assertEqual(order_a,order_c)
        self.assertEqual(m.order_qa_clusters(permuted_b)[0]['video'],2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=self.write_test_images(root,units_a)
            out_a=root/'run_a';out_a.mkdir()
            out_b=root/'run_b';out_b.mkdir()
            saved_seq_a=[];saved_seq_b=[]
            orig_load=m.load_source_rgb
            def trace_a(src,meta):
                saved_seq_a.append(meta['id'])
                return orig_load(src,meta)
            def trace_b(src,meta):
                saved_seq_b.append(meta['id'])
                return orig_load(src,meta)
            with patch.object(m,'load_source_rgb',side_effect=trace_a):
                ra=m.identity_qa_panels(out_a,clusters_a,units_a,images,root,max_panels=10)
            with patch.object(m,'load_source_rgb',side_effect=trace_b):
                rb=m.identity_qa_panels(out_b,permuted_b,units_b,images,root,max_panels=10)
            self.assertEqual(ra,5)
            self.assertEqual(rb,5)
            self.assertEqual(saved_seq_a,saved_seq_b)
            for cid in order_a:
                ba=(out_a/'identity_qa_panels'/f'{cid}.png').read_bytes()
                bb=(out_b/'identity_qa_panels'/f'{cid}.png').read_bytes()
                self.assertEqual(hashlib.sha256(ba).hexdigest(),hashlib.sha256(bb).hexdigest())

    @staticmethod
    def synthetic_build_env(root):
        # Build a synthetic 248-GT-bearing-image, 1025-GT-unit environment producing >240 provisional clusters
        from types import SimpleNamespace
        zpath,metas=ZipNativeV5Contracts.fixture(root,n=248,include_test=True)
        slots=[(v,f) for v in m.TRAIN if v!=31 for f in range(0,160,10) if not (v==8 and f<80)]
        assert len(slots)==248
        imgs={};anns={};rows=[]
        slot_iids=[]
        for (vid,frame),iid in zip(slots,sorted(metas)):
            meta=dict(metas[iid])
            meta['video_id']=vid
            meta['frame_id']=frame
            meta['_p2588_cached_frame_idx']=frame
            imgs[iid]=meta
            slot_iids.append((vid,frame,iid))
        # Alternate classes across consecutive frames ((frame//10 + pass_idx) % 7) so consecutive frames
        # have distinct classes and form >= 248 provisional clusters, directly testing the old 120 cap defect.
        for aid in range(1,1026):
            slot_idx=(aid-1)%248
            vid,frame,iid=slot_iids[slot_idx]
            pass_idx=(aid-1)//248
            cl=((frame//10)+pass_idx)%7
            x=float(1+(pass_idx*5)%20)
            y=float(1+(pass_idx*3)%14)
            anns[aid]={'id':aid,'image_id':iid,'category_id':cl+1,'bbox':[x,y,4.0,4.0]}
            rows.append({'annotation_id':str(aid),'image_id':str(iid),'video':str(vid),
                         'frame':str(frame),'class':str(cl),'class_name':m.CLASS_NAMES[cl],
                         'is_tp':str(aid%2),'gt_index':str(pass_idx),'gt_area_source_px2':'16.0',
                         'best_raw_query':'1','signed_center_x_over_gt_w':'0.0',
                         'signed_center_y_over_gt_h':'0.0','log_pred_w_over_gt_w':'0.0',
                         'log_pred_h_over_gt_h':'0.0'})
        coco_path=root/'seaclips-train.json'
        coco_path.write_text(json.dumps({
            'categories':[{'id':i+1,'name':n} for i,n in enumerate(m.CLASS_NAMES)],
            'images':list(imgs.values()),
            'annotations':list(anns.values())
        }),encoding='utf-8')
        allow_imgs={iid:[] for iid in sorted(metas)}
        for r in rows:
            allow_imgs[int(r['image_id'])].append(r)
        pins={'coco':{'path':str(coco_path),'sha256':'synthetic'},
              'csv':{'path':'synthetic.csv','sha256':'synthetic'},
              'spatial':{'path':'spatial.json','sha256':'synthetic'},
              'fn':{'path':'fn.json','sha256':'synthetic'}}
        args=SimpleNamespace(
            phase='build',project=str(root),coco_train=str(coco_path),
            output=str(root/'reviewer_out'),images_root=None,images_zip=str(zpath),
            sealed_root=str(root/'private_out'),max_qa_panels=1025,identity_review=None,
            authorize_stage1_build=True
        )
        return args,pins,rows,allow_imgs,imgs,anns

    def test_build_and_preflight_reject_insufficient_cap_including_legacy_120_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=self.synthetic_build_env(root)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                args.phase='preflight'
                args.max_qa_panels=120
                with self.assertRaisesRegex(AssertionError,r'max_qa_panels \(120\) must be >= cluster_count'):
                    m.build(args)
                args.phase='build'
                with self.assertRaisesRegex(AssertionError,r'max_qa_panels \(120\) must be >= cluster_count'):
                    m.build(args)
                self.assertFalse((root/'reviewer_out').exists())
                self.assertFalse((root/'private_out').exists())

    def test_build_fails_closed_and_cleans_staging_if_rendered_count_differs_from_cluster_count(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=self.synthetic_build_env(root)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                 patch.object(m,'identity_qa_panels',return_value=120):
                with self.assertRaisesRegex(AssertionError,'rendered_panel_count .* != cluster_count'):
                    m.build(args)
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build').exists())
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

    def test_build_succeeds_when_all_clusters_are_rendered_and_verifies_exact_count(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=self.synthetic_build_env(root)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.build(args)
            build_dir=root/'reviewer_out'/'P2588_identity_build'
            private_dir=root/'private_out'/'P2588_sealed_cluster_outcomes'
            self.assertTrue(build_dir.is_dir())
            self.assertTrue(private_dir.is_dir())
            rep=json.loads((build_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
            n_clusters=rep['n_provisional_clusters_NOT_independent_objects']
            self.assertGreaterEqual(n_clusters,240)
            self.assertEqual(rep['identity_QA_panels_rendered'],n_clusters)
            self.assertTrue(rep['identity_QA_panel_coverage_complete'])
            self.assertEqual(rep['identity_QA_panel_ordering'],'OUTCOME_INDEPENDENT_PROVISIONAL_CLUSTER_ID_ASC')
            self.assertEqual(len(list((build_dir/'identity_qa_panels').glob('*.png'))),n_clusters)

    def test_direct_adjudicate_call_is_blocked_on_hold(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(AssertionError,'Stage 2 human adjudication on HOLD'):
            m.adjudicate(SimpleNamespace())

    def test_v6_installer_and_rollback_preserve_v5_and_refuse_existing_outputs(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if (here.parent/'APPLY_P2588_ZIP_NATIVE_V6.py').is_file() else (here/'OCT15_P2588_ZIP_NATIVE_V6' if (here/'OCT15_P2588_ZIP_NATIVE_V6'/'APPLY_P2588_ZIP_NATIVE_V6.py').is_file() else here.parent)
        inst_file=next((pkg_root/n for n in ('APPLY_P2588_ZIP_NATIVE_V6.py','APPLY_P2588_ZIP_NATIVE_V7.py','APPLY_P2588_ZIP_NATIVE_V8.py','APPLY_P2588_ZIP_NATIVE_V9.py','APPLY_P2588_ZIP_NATIVE_V10.py','APPLY_P2588_ZIP_NATIVE_V11.py','APPLY_P2588_ZIP_NATIVE_V12.py') if (pkg_root/n).is_file()),pkg_root/'APPLY_P2588_ZIP_NATIVE_V8.py')
        rb_file=next((pkg_root/n for n in ('ROLLBACK_P2588_V6.py','ROLLBACK_P2588_V7.py','ROLLBACK_P2588_V8.py','ROLLBACK_P2588_V9.py','ROLLBACK_P2588_V10.py','ROLLBACK_P2588_V11.py','ROLLBACK_P2588_V12.py') if (pkg_root/n).is_file()),pkg_root/'ROLLBACK_P2588_V8.py')
        if not inst_file.is_file():
            self.skipTest('Package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v6',inst_file)
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v6',rb_file)
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            # Populate mock V5 files with exact V5 content if available, or test via mocked manifest
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v5_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v5_test')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':manifest['schema'],
                'previous_sources':{'v5':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            # Repeat install over V6 must fail closed at source-pin gate
            with self.assertRaisesRegex(AssertionError,'Expected exactly pinned installed'):
                apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))
            # Rollback must restore exact V5 bytes
            rb_mod.rollback(proj,fake_pkg)
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v5_runner')
            # Repeat install after rollback when backup dir already exists must also fail closed
            with self.assertRaisesRegex(AssertionError,'repeat installation refused'):
                apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))


class V7TransactionalPublicationAndPrivacyContracts(unittest.TestCase):
    def test_second_publication_failure_rolls_back_first_published_tree_and_all_staging(self):
        # Experiment Request 1: Inject failure after private publication succeeds but before reviewer publication;
        # verify that both final destinations and both staging directories are absent afterward.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            calls=[]
            orig_pub=m.publish_staged_output
            def fail_on_second(tmp,dest):
                calls.append(dest)
                if len(calls)==2:
                    raise RuntimeError('Simulated failure during second publication')
                return orig_pub(tmp,dest)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                 patch.object(m,'publish_staged_output',side_effect=fail_on_second):
                with self.assertRaisesRegex(RuntimeError,'Simulated failure during second publication'):
                    m.build(args)
            self.assertEqual(len(calls),2)
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build').exists())
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

    def test_second_staging_creation_failure_cleans_up_first_staging_directory(self):
        # Experiment Request 2: Inject failure while creating the second staging directory;
        # verify cleanup of the first staging directory.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            calls=[]
            orig_tmp=m.tmp_output
            def fail_second_tmp(dest):
                calls.append(dest)
                if len(calls)==2:
                    raise RuntimeError('Simulated failure creating second staging directory')
                return orig_tmp(dest)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                 patch.object(m,'tmp_output',side_effect=fail_second_tmp):
                with self.assertRaisesRegex(RuntimeError,'Simulated failure creating second staging directory'):
                    m.build(args)
            self.assertEqual(len(calls),2)
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build').exists())
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

    def test_preexisting_outputs_are_never_deleted_when_build_refuses(self):
        # Ensure write-once protection refuses before transaction and never deletes pre-existing directories
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            pre_priv=root/'private_out'/'P2588_sealed_cluster_outcomes'
            pre_priv.mkdir(parents=True)
            sentinel=pre_priv/'SENTINEL.txt';sentinel.write_text('KEEP',encoding='utf-8')
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                with self.assertRaisesRegex(AssertionError,'already exists'):
                    m.build(args)
            self.assertTrue(sentinel.is_file())
            self.assertEqual(sentinel.read_text(encoding='utf-8'),'KEEP')

    def test_reviewer_package_recursive_privacy_and_source_path_redaction_audit(self):
        # Experiment Request 3: Build a synthetic reviewer package and recursively audit its contents
        # for source paths, outcome columns, and prediction-derived fields.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,_,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            secret_paths={
                'coco':{'path':str(root/'SECRET_DIR'/'seaclips-train.json'),'sha256':m.TRAIN_COCO_SHA},
                'csv':{'path':str(root/'SECRET_DIR'/'train_spatial_units.csv'),'sha256':m.SPATIAL_CSV_SHA},
                'spatial':{'path':str(root/'SECRET_DIR'/'train_spatial_diagnostic.json'),'sha256':m.SPATIAL_JSON_SHA},
                'fn':{'path':str(root/'SECRET_DIR'/'train_fn_recoverability.json'),'sha256':m.FN_JSON_SHA},
            }
            with patch.object(m,'verify_sources',return_value=secret_paths), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.build(args)
            build_dir=root/'reviewer_out'/'P2588_identity_build'
            private_dir=root/'private_out'/'P2588_sealed_cluster_outcomes'
            rep=json.loads((build_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
            # Reviewer report source_pins must contain source_role and sha256 only, no 'path'
            for k,v in rep['source_pins'].items():
                self.assertEqual(set(v.keys()),{'source_role','sha256'})
                self.assertNotIn('path',v)
            self.assertNotIn('image_zip',rep['train_image_header_verification'])
            self.assertNotIn('image_root',rep['train_image_header_verification'])
            # Private SEALED_README.json retains full path provenance
            sealed_readme=json.loads((private_dir/'SEALED_README.json').read_text(encoding='utf-8'))
            self.assertEqual(sealed_readme['source_pins']['csv']['path'],secret_paths['csv']['path'])
            self.assertTrue(sealed_readme['image_zip_path'].endswith('SeaClips.zip'))
            # Recursively scan all non-binary files in reviewer package
            forbidden_substrings=(
                str(root), 'SECRET_DIR', 'train_spatial_units.csv',
                'train_spatial_diagnostic.json', 'train_fn_recoverability.json',
                'SeaClips.zip', 'is_tp', 'n_tp', 'n_fn', 'best_raw_query',
                'signed_center_x_over_gt_w', 'signed_center_y_over_gt_h',
                'log_pred_w_over_gt_w', 'log_pred_h_over_gt_h'
            )
            for fpath in build_dir.rglob('*'):
                if fpath.is_file() and fpath.suffix in ('.csv','.json','.md'):
                    # Placeholder empty CSV headers in per_object_persistence.csv have fn_frames/fn_fraction header only
                    content=fpath.read_text(encoding='utf-8')
                    for bad in forbidden_substrings:
                        self.assertNotIn(bad,content,f'Forbidden disclosure {bad!r} in {fpath.name}')
                elif fpath.is_file() and fpath.suffix=='.png':
                    from PIL import Image
                    with Image.open(fpath) as pim:
                        meta_str=json.dumps({str(k):str(v) for k,v in pim.info.items()})
                        for bad in forbidden_substrings:
                            self.assertNotIn(bad,meta_str,f'Forbidden disclosure {bad!r} in PNG metadata {fpath.name}')

    def test_malformed_or_path_traversing_cluster_ids_rejected_before_file_creation(self):
        # Experiment Request 4: Test rejection of malformed or path-traversing provisional cluster IDs.
        clusters,units=V6CompleteOutcomeIndependentQAPanelsContracts.make_clusters_and_units()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=V6CompleteOutcomeIndependentQAPanelsContracts.write_test_images(root,units)
            for bad_id in ('../P0001','P0001/evil','P1','P0000','CLUSTER_1','P0001.png','C:P0001'):
                out=root/f'out_{abs(hash(bad_id))}';out.mkdir(exist_ok=True)
                bad_clusters=[{**clusters[0],'provisional_cluster_id':bad_id},*clusters[1:]]
                with self.assertRaisesRegex(AssertionError,'Malformed provisional_cluster_id'):
                    m.identity_qa_panels(out,bad_clusters,units,images,root,max_panels=10)
                self.assertFalse((out/'identity_qa_panels').exists())

    def test_stage1_build_requires_explicit_director_authorization_flag_in_main(self):
        with patch.object(sys,'argv',['runner','--phase','build','--sealed-root','/tmp/sealed']):
            with self.assertRaisesRegex(AssertionError,'Stage 1 build is on HOLD; pass --authorize-stage1-build'):
                m.main()

    def test_v7_installer_and_rollback_accept_v6_and_preserve_write_once(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if (here.parent/'APPLY_P2588_ZIP_NATIVE_V7.py').is_file() else (here/'OCT15_P2588_ZIP_NATIVE_V7' if (here/'OCT15_P2588_ZIP_NATIVE_V7'/'APPLY_P2588_ZIP_NATIVE_V7.py').is_file() else here.parent)
        inst_file=next((pkg_root/n for n in ('APPLY_P2588_ZIP_NATIVE_V7.py','APPLY_P2588_ZIP_NATIVE_V8.py','APPLY_P2588_ZIP_NATIVE_V9.py','APPLY_P2588_ZIP_NATIVE_V10.py','APPLY_P2588_ZIP_NATIVE_V11.py','APPLY_P2588_ZIP_NATIVE_V12.py') if (pkg_root/n).is_file()),pkg_root/'APPLY_P2588_ZIP_NATIVE_V9.py')
        rb_file=next((pkg_root/n for n in ('ROLLBACK_P2588_V7.py','ROLLBACK_P2588_V8.py','ROLLBACK_P2588_V9.py','ROLLBACK_P2588_V10.py','ROLLBACK_P2588_V11.py','ROLLBACK_P2588_V12.py') if (pkg_root/n).is_file()),pkg_root/'ROLLBACK_P2588_V9.py')
        if not inst_file.is_file():
            self.skipTest('V7 package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v7',inst_file)
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v7',rb_file)
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v6_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v6_test')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':manifest['schema'],
                'previous_sources':{'v6':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            rb_mod.rollback(proj,fake_pkg)
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v6_runner')


class V8PostRenameRollbackAndDirectBuildAuthContracts(unittest.TestCase):
    def test_post_mkdir_exception_in_second_staging_creation_rolls_back_both_staging_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            calls=[]
            orig_tmp=m.tmp_output
            def fail_after_second_mkdir(dest):
                calls.append(dest)
                created=orig_tmp(dest)
                self.assertTrue(Path(created).is_dir())
                if len(calls)==2:
                    raise RuntimeError('Simulated post-mkdir failure in second tmp_output')
                return created
            with patch.object(m,'verify_sources',return_value=pins),                  patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})),                  patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)),                  patch.object(m,'tmp_output',side_effect=fail_after_second_mkdir):
                with self.assertRaisesRegex(RuntimeError,'Simulated post-mkdir failure in second tmp_output'):
                    m.build(args)
            self.assertEqual(len(calls),2)
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

    def test_post_rename_exception_in_second_publication_rolls_back_both_final_trees(self):
        # Experiment Request 1: Let the second publish_staged_output() rename succeed, then inject an exception.
        # Verify that both final destinations and both staging directories are absent afterward.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            calls=[]
            orig_pub=m.publish_staged_output
            def fail_after_second_rename(tmp,dest):
                calls.append(dest)
                # Invoke the real publication function so tmp.rename(dest) succeeds on disk!
                orig_pub(tmp,dest)
                self.assertTrue(Path(dest).is_dir())
                if len(calls)==2:
                    raise RuntimeError('Simulated post-rename failure after second publication')
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                 patch.object(m,'publish_staged_output',side_effect=fail_after_second_rename):
                with self.assertRaisesRegex(RuntimeError,'Simulated post-rename failure after second publication'):
                    m.build(args)
            self.assertEqual(len(calls),2)
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build').exists())
            self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

    def test_rollback_cleanup_failure_is_reported_explicitly(self):
        # Experiment Request 2: Inject a cleanup failure and verify that the build reports rollback failure
        # explicitly rather than implying the transaction was fully reversed.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            orig_pub=m.publish_staged_output
            calls=[]
            def fail_after_first_rename(tmp,dest):
                calls.append(dest)
                orig_pub(tmp,dest)
                raise RuntimeError('Primary publication error')
            def broken_rmtree(path,*a,**kw):
                raise PermissionError(f'Simulated OS lock on {path}')
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                 patch.object(m,'publish_staged_output',side_effect=fail_after_first_rename), \
                 patch.object(m.shutil,'rmtree',side_effect=broken_rmtree):
                with self.assertRaisesRegex(RuntimeError,r'\[P2588 ROLLBACK FAILURE\] Failed to remove owned outputs'):
                    m.build(args)

    def test_direct_build_call_without_authorization_fails_before_creating_outputs(self):
        # Experiment Request 3: Call build() directly with build phase and no authorization acknowledgement.
        # It must fail before creating any directory or file.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            args.authorize_stage1_build=False
            with self.assertRaisesRegex(AssertionError,'Stage 1 build is on HOLD; pass --authorize-stage1-build'):
                m.build(args)
            self.assertFalse((root/'reviewer_out').exists())
            self.assertFalse((root/'private_out').exists())
            delattr(args,'authorize_stage1_build')
            with self.assertRaisesRegex(AssertionError,'Stage 1 build is on HOLD; pass --authorize-stage1-build'):
                m.build(args)
            self.assertFalse((root/'reviewer_out').exists())
            self.assertFalse((root/'private_out').exists())

    def test_v8_installer_and_rollback_accept_v7_and_preserve_write_once(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if (here.parent/'APPLY_P2588_ZIP_NATIVE_V8.py').is_file() else (here/'OCT15_P2588_ZIP_NATIVE_V8' if (here/'OCT15_P2588_ZIP_NATIVE_V8'/'APPLY_P2588_ZIP_NATIVE_V8.py').is_file() else here.parent)
        inst_file=next((pkg_root/n for n in ('APPLY_P2588_ZIP_NATIVE_V8.py','APPLY_P2588_ZIP_NATIVE_V9.py','APPLY_P2588_ZIP_NATIVE_V10.py','APPLY_P2588_ZIP_NATIVE_V11.py','APPLY_P2588_ZIP_NATIVE_V12.py') if (pkg_root/n).is_file()),pkg_root/'APPLY_P2588_ZIP_NATIVE_V9.py')
        rb_file=next((pkg_root/n for n in ('ROLLBACK_P2588_V8.py','ROLLBACK_P2588_V9.py','ROLLBACK_P2588_V10.py','ROLLBACK_P2588_V11.py','ROLLBACK_P2588_V12.py') if (pkg_root/n).is_file()),pkg_root/'ROLLBACK_P2588_V9.py')
        if not inst_file.is_file():
            self.skipTest('V8 package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v8',inst_file)
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v8',rb_file)
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v7_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v7_test')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':manifest['schema'],
                'previous_sources':{'v7':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            rb_mod.rollback(proj,fake_pkg)
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v7_runner')


class V9StagingInitManifestAndCleanupVerificationContracts(unittest.TestCase):
    def test_initial_manifest_write_failure_cleans_up_staging_directory_on_first_or_second_init(self):
        # Experiment Request 1: Inject an initial ownership-manifest write failure and confirm
        # successful cleanup removes the staging directory (tested on both 1st and 2nd staging init).
        for fail_call_index in (1,2):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td)
                args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
                orig_write=m.write_invocation_manifest
                calls=[]
                def fail_manifest_write(folder,files,invocation_token=None):
                    calls.append(Path(folder))
                    if len(calls)==fail_call_index:
                        raise OSError(f'Simulated initial manifest write failure on call {fail_call_index}')
                    return orig_write(folder,files,invocation_token=invocation_token)
                with patch.object(m,'verify_sources',return_value=pins), \
                     patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                     patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                     patch.object(m,'write_invocation_manifest',side_effect=fail_manifest_write):
                    with self.assertRaisesRegex(OSError,f'Simulated initial manifest write failure on call {fail_call_index}'):
                        m.build(args)
                self.assertEqual(len(calls),fail_call_index)
                self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())
                self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())
                self.assertFalse((root/'reviewer_out'/'P2588_identity_build').exists())
                self.assertFalse((root/'private_out'/'P2588_sealed_cluster_outcomes').exists())

    def test_initial_manifest_write_failure_plus_cleanup_failure_raises_explicit_rollback_failure_identifying_path(self):
        # Experiment Request 2: Inject the same manifest-write failure plus a cleanup failure;
        # require explicit [P2588 ROLLBACK FAILURE] reporting and verify the remaining path is identified.
        for fail_call_index,expected_leaf in ((1,'P2588_identity_build.staging'),(2,'P2588_sealed_cluster_outcomes.staging')):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td)
                args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
                orig_write=m.write_invocation_manifest
                orig_rmtree=m.shutil.rmtree
                calls=[]
                def fail_manifest_write(folder,files,invocation_token=None):
                    calls.append(Path(folder))
                    if len(calls)==fail_call_index:
                        raise OSError(f'Simulated manifest write error on {folder}')
                    return orig_write(folder,files,invocation_token=invocation_token)
                def fail_rmtree_for_target(path,*a,**kw):
                    if Path(path).name==expected_leaf:
                        raise PermissionError(f'Simulated cleanup lock on {path}')
                    return orig_rmtree(path,*a,**kw)
                with patch.object(m,'verify_sources',return_value=pins), \
                     patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                     patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                     patch.object(m,'write_invocation_manifest',side_effect=fail_manifest_write), \
                     patch.object(m.shutil,'rmtree',side_effect=fail_rmtree_for_target):
                    with self.assertRaises(RuntimeError) as ctx:
                        m.build(args)
                msg=str(ctx.exception)
                self.assertIn('[P2588 ROLLBACK FAILURE]',msg)
                self.assertIn(expected_leaf,msg)
                self.assertIsInstance(ctx.exception.__cause__,OSError)
                if fail_call_index==2:
                    # First staging directory was still rolled back cleanly even when second failed cleanup!
                    self.assertFalse((root/'reviewer_out'/'P2588_identity_build.staging').exists())

    def test_direct_tmp_output_manifest_write_failure_verifies_cleanup_and_reports_cleanup_error(self):
        # Direct unit test of tmp_output() when ACTIVE_INVOCATION_TOKEN is None or set:
        # 1) Manifest write fails + cleanup succeeds -> staging removed.
        # 2) Manifest write fails + cleanup fails -> explicit [P2588 ROLLBACK FAILURE] naming path.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            dest1=root/'out1'/'P2588_identity_build'
            dest1.parent.mkdir(parents=True)
            with patch.object(m,'write_invocation_manifest',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):
                    m.tmp_output(dest1)
            self.assertFalse(Path(str(dest1)+'.staging').exists())
            dest2=root/'out2'/'P2588_identity_build'
            dest2.parent.mkdir(parents=True)
            with patch.object(m,'write_invocation_manifest',side_effect=OSError('disk full')), \
                 patch.object(m.shutil,'rmtree',side_effect=PermissionError('locked')):
                with self.assertRaises(RuntimeError) as ctx:
                    m.tmp_output(dest2)
            self.assertIn('[P2588 ROLLBACK FAILURE]',str(ctx.exception))
            self.assertIn(str(Path(str(dest2)+'.staging')),str(ctx.exception))

    def test_silent_noop_manifest_write_is_rejected_and_cleaned_up_or_reports_rollback_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            dest1=root/'out1'/'P2588_identity_build'
            dest1.parent.mkdir(parents=True)
            with patch.object(m,'write_invocation_manifest',return_value=None):
                with self.assertRaisesRegex(AssertionError,'Staging ownership manifest verification failed'):
                    m.tmp_output(dest1)
            self.assertFalse(Path(str(dest1)+'.staging').exists())
            dest2=root/'out2'/'P2588_identity_build'
            dest2.parent.mkdir(parents=True)
            with patch.object(m,'write_invocation_manifest',return_value=None),                  patch.object(m.shutil,'rmtree',side_effect=PermissionError('locked on noop cleanup')):
                with self.assertRaises(RuntimeError) as ctx:
                    m.tmp_output(dest2)
            self.assertIn('[P2588 ROLLBACK FAILURE]',str(ctx.exception))
            self.assertIn(str(Path(str(dest2)+'.staging')),str(ctx.exception))

    def test_v9_installer_and_rollback_accept_v8_and_preserve_write_once(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if (here.parent/'APPLY_P2588_ZIP_NATIVE_V9.py').is_file() else (here/'OCT15_P2588_ZIP_NATIVE_V9' if (here/'OCT15_P2588_ZIP_NATIVE_V9'/'APPLY_P2588_ZIP_NATIVE_V9.py').is_file() else here.parent)
        inst_file=next((pkg_root/n for n in ('APPLY_P2588_ZIP_NATIVE_V9.py','APPLY_P2588_ZIP_NATIVE_V10.py','APPLY_P2588_ZIP_NATIVE_V11.py','APPLY_P2588_ZIP_NATIVE_V12.py') if (pkg_root/n).is_file()),pkg_root/'APPLY_P2588_ZIP_NATIVE_V10.py')
        rb_file=next((pkg_root/n for n in ('ROLLBACK_P2588_V9.py','ROLLBACK_P2588_V10.py','ROLLBACK_P2588_V11.py','ROLLBACK_P2588_V12.py') if (pkg_root/n).is_file()),pkg_root/'ROLLBACK_P2588_V10.py')
        if not inst_file.is_file():
            self.skipTest('V9 package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v9',inst_file)
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v9',rb_file)
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v8_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v8_test')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':manifest['schema'],
                'previous_sources':{'v8':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            rb_mod.rollback(proj,fake_pkg)
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v8_runner')


class V10StagingRegistrationFailureAndRollbackContracts(unittest.TestCase):
    def test_staging_registry_append_failure_after_mkdir_cleans_up_staging_directory(self):
        # Experiment Request 1: Inject an exception from ACTIVE_CREATED_STAGING.append() after mkdir() succeeds;
        # verify that the staging directory is removed (tested on direct tmp_output and in build on 1st & 2nd init).
        class FailingAppendList(list):
            def __init__(self,fail_on_call=1):
                super().__init__()
                self.calls=0
                self.fail_on_call=fail_on_call
            def append(self,item):
                self.calls+=1
                if self.calls==self.fail_on_call:
                    raise MemoryError(f'Simulated registry append failure on call {self.calls} for {item}')
                return super().append(item)

        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            # 1a. Direct tmp_output() test with failing ACTIVE_CREATED_STAGING.append()
            dest_direct=root/'direct_out'/'P2588_identity_build'
            dest_direct.parent.mkdir(parents=True)
            failing_reg=FailingAppendList(fail_on_call=1)
            with patch.object(m,'ACTIVE_CREATED_STAGING',failing_reg), \
                 patch.object(m,'ACTIVE_INVOCATION_TOKEN','tok123'):
                with self.assertRaisesRegex(MemoryError,'Simulated registry append failure on call 1'):
                    m.tmp_output(dest_direct)
            self.assertFalse(Path(str(dest_direct)+'.staging').exists())
            self.assertEqual(len(failing_reg),0)

            # 1b. Full build() test when 1st or 2nd staging registration fails
            for fail_idx in (1,2):
                sub=root/f'build_case_{fail_idx}'
                sub.mkdir(parents=True)
                args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(sub)
                orig_tmp_output=m.tmp_output
                def tmp_with_failing_registry(dest,*,_fail_idx=fail_idx):
                    if isinstance(m.ACTIVE_CREATED_STAGING,list) and not isinstance(m.ACTIVE_CREATED_STAGING,FailingAppendList):
                        w=FailingAppendList(fail_on_call=_fail_idx)
                        w.extend(m.ACTIVE_CREATED_STAGING)
                        m.ACTIVE_CREATED_STAGING=w
                    return orig_tmp_output(dest)
                with patch.object(m,'verify_sources',return_value=pins), \
                     patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                     patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                     patch.object(m,'tmp_output',side_effect=tmp_with_failing_registry):
                    with self.assertRaisesRegex(MemoryError,f'Simulated registry append failure on call {fail_idx}'):
                        m.build(args)
                self.assertFalse((sub/'reviewer_out'/'P2588_identity_build.staging').exists())
                self.assertFalse((sub/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())
                self.assertFalse((sub/'reviewer_out'/'P2588_identity_build').exists())
                self.assertFalse((sub/'private_out'/'P2588_sealed_cluster_outcomes').exists())

    def test_staging_registry_append_failure_plus_cleanup_failure_raises_explicit_rollback_failure_identifying_path(self):
        # Experiment Request 2: Inject the same registration failure plus a cleanup failure;
        # require explicit [P2588 ROLLBACK FAILURE] reporting and identification of the remaining path.
        class FailingAppendList(list):
            def __init__(self,fail_on_call=1):
                super().__init__()
                self.calls=0
                self.fail_on_call=fail_on_call
            def append(self,item):
                self.calls+=1
                if self.calls==self.fail_on_call:
                    raise MemoryError(f'Simulated registry append failure on call {self.calls} for {item}')
                return super().append(item)

        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            # 2a. Direct tmp_output() with registration failure + cleanup failure
            dest_direct=root/'direct_fail'/'P2588_identity_build'
            dest_direct.parent.mkdir(parents=True)
            failing_reg=FailingAppendList(fail_on_call=1)
            with patch.object(m,'ACTIVE_CREATED_STAGING',failing_reg), \
                 patch.object(m,'ACTIVE_INVOCATION_TOKEN','tok123'), \
                 patch.object(m.shutil,'rmtree',side_effect=PermissionError('Simulated rmtree lock during reg failure')):
                with self.assertRaises(RuntimeError) as ctx:
                    m.tmp_output(dest_direct)
            msg=str(ctx.exception)
            self.assertIn('[P2588 ROLLBACK FAILURE]',msg)
            self.assertIn(str(Path(str(dest_direct)+'.staging')),msg)
            self.assertIsInstance(ctx.exception.__cause__,MemoryError)

            # 2b. Full build() with registration failure + cleanup failure on 1st or 2nd staging init
            for fail_idx,expected_leaf in ((1,'P2588_identity_build.staging'),(2,'P2588_sealed_cluster_outcomes.staging')):
                sub=root/f'build_reg_fail_{fail_idx}'
                sub.mkdir(parents=True)
                args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(sub)
                orig_tmp_output=m.tmp_output
                orig_rmtree=m.shutil.rmtree
                def tmp_with_failing_registry(dest,*,_fail_idx=fail_idx):
                    if isinstance(m.ACTIVE_CREATED_STAGING,list) and not isinstance(m.ACTIVE_CREATED_STAGING,FailingAppendList):
                        w=FailingAppendList(fail_on_call=_fail_idx)
                        w.extend(m.ACTIVE_CREATED_STAGING)
                        m.ACTIVE_CREATED_STAGING=w
                    return orig_tmp_output(dest)
                def fail_rmtree_for_leaf(path,*a,**kw):
                    if Path(path).name==expected_leaf:
                        raise PermissionError(f'Simulated rmtree lock on {path}')
                    return orig_rmtree(path,*a,**kw)
                with patch.object(m,'verify_sources',return_value=pins), \
                     patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                     patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                     patch.object(m,'tmp_output',side_effect=tmp_with_failing_registry), \
                     patch.object(m.shutil,'rmtree',side_effect=fail_rmtree_for_leaf):
                    with self.assertRaises(RuntimeError) as ctx2:
                        m.build(args)
                msg2=str(ctx2.exception)
                self.assertIn('[P2588 ROLLBACK FAILURE]',msg2)
                self.assertIn(expected_leaf,msg2)
                self.assertIsInstance(ctx2.exception.__cause__,MemoryError)
                if fail_idx==2:
                    # 1st staging directory was still rolled back cleanly when 2nd failed registration + cleanup!
                    self.assertFalse((sub/'reviewer_out'/'P2588_identity_build.staging').exists())

    def test_published_final_registry_append_failure_after_rename_cleans_up_or_reports_rollback_failure(self):
        # Verify publish_staged_output() also protects post-rename registration in ACTIVE_PUBLISHED_FINAL.
        class FailingPubList(list):
            def append(self,item):
                raise MemoryError(f'Simulated published registry append failure for {item}')

        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            tmp1=root/'out1.staging';tmp1.mkdir()
            (tmp1/'file.txt').write_text('ok',encoding='utf-8')
            dest1=root/'out1'
            with patch.object(m,'ACTIVE_PUBLISHED_FINAL',FailingPubList()):
                with self.assertRaisesRegex(MemoryError,'Simulated published registry append failure'):
                    m.publish_staged_output(tmp1,dest1)
            self.assertFalse(tmp1.exists())
            self.assertFalse(dest1.exists())

            tmp2=root/'out2.staging';tmp2.mkdir()
            (tmp2/'file.txt').write_text('ok',encoding='utf-8')
            dest2=root/'out2'
            with patch.object(m,'ACTIVE_PUBLISHED_FINAL',FailingPubList()), \
                 patch.object(m.shutil,'rmtree',side_effect=PermissionError('locked final')):
                with self.assertRaises(RuntimeError) as ctx:
                    m.publish_staged_output(tmp2,dest2)
            self.assertIn('[P2588 ROLLBACK FAILURE]',str(ctx.exception))
            self.assertIn(str(dest2),str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__,MemoryError)

    def test_v10_installer_and_rollback_accept_v9_and_preserve_write_once(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if (here.parent/'APPLY_P2588_ZIP_NATIVE_V10.py').is_file() else (here/'OCT15_P2588_ZIP_NATIVE_V10' if (here/'OCT15_P2588_ZIP_NATIVE_V10'/'APPLY_P2588_ZIP_NATIVE_V10.py').is_file() else here.parent)
        inst_file=next((pkg_root/n for n in ('APPLY_P2588_ZIP_NATIVE_V10.py','APPLY_P2588_ZIP_NATIVE_V11.py','APPLY_P2588_ZIP_NATIVE_V12.py') if (pkg_root/n).is_file()),pkg_root/'APPLY_P2588_ZIP_NATIVE_V11.py')
        rb_file=next((pkg_root/n for n in ('ROLLBACK_P2588_V10.py','ROLLBACK_P2588_V11.py','ROLLBACK_P2588_V12.py') if (pkg_root/n).is_file()),pkg_root/'ROLLBACK_P2588_V11.py')
        if not inst_file.is_file():
            self.skipTest('V10 package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v10',inst_file)
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v10',rb_file)
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v9_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v9_test')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':manifest['schema'],
                'previous_sources':{'v9':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            apply_mod.apply(proj,fake_pkg,sealed_root=str(Path(td)/'private'))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            rb_mod.rollback(proj,fake_pkg)
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v9_runner')



class V11ProvenancePairVerifyAndHardeningContracts(unittest.TestCase):
    def test_runner_sha256_embedded_in_both_manifests_report_and_sealed_readme(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.build(args)
            rev_dir=root/'reviewer_out'/'P2588_identity_build'
            priv_dir=root/'private_out'/'P2588_sealed_cluster_outcomes'
            expected_rsha=m.current_runner_sha256()
            self.assertEqual(len(expected_rsha),64)
            rev_man=json.loads((rev_dir/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
            priv_man=json.loads((priv_dir/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
            rep=json.loads((rev_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
            sealed_readme=json.loads((priv_dir/'SEALED_README.json').read_text(encoding='utf-8'))
            self.assertEqual(rev_man['runner_sha256'],expected_rsha)
            self.assertEqual(priv_man['runner_sha256'],expected_rsha)
            self.assertEqual(rep['runner_sha256'],expected_rsha)
            self.assertEqual(sealed_readme['runner_sha256'],expected_rsha)
            self.assertEqual(rev_man['invocation_token'],priv_man['invocation_token'])

    def test_verify_build_phase_validates_two_tree_pair_and_detects_tampering_or_token_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.build(args)
            rev_dir=root/'reviewer_out'/'P2588_identity_build'
            priv_dir=root/'private_out'/'P2588_sealed_cluster_outcomes'
            res=m.verify_published_build_pair(rev_dir,priv_dir,expected_runner_sha=m.current_runner_sha256())
            self.assertTrue(res['both_trees_paired_and_verified'])
            # CLI --phase verify-build also succeeds read-only
            with patch.object(sys,'argv',['runner','--phase','verify-build','--output',str(root/'reviewer_out'),'--sealed-root',str(root/'private_out')]):
                m.main()
            # Extra file in reviewer tree must fail verification
            extra=rev_dir/'UNEXPECTED.txt'
            extra.write_text('extra',encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'On-disk file set differs'):
                m.verify_published_build_pair(rev_dir,priv_dir)
            extra.unlink()
            # Mismatched invocation_token between reviewer and private trees must fail verification
            priv_mpath=priv_dir/'SHA256_MANIFEST.json'
            orig_priv_man=priv_mpath.read_text(encoding='utf-8')
            tampered=json.loads(orig_priv_man)
            tampered['invocation_token']='0'*32
            priv_mpath.write_text(json.dumps(tampered),encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'invocation_token mismatch'):
                m.verify_published_build_pair(rev_dir,priv_dir)
            priv_mpath.write_text(orig_priv_man,encoding='utf-8')

    def test_write_trace_redacts_drive_letter_paths_and_outcome_filenames_from_real_project_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            proj=root/'proj';proj.mkdir()
            out=root/'out';out.mkdir()
            (proj/'train_bfcef_f_s0_final_experts.py').write_text(
                'def build_transform():\n'
                '    # resize and normalize from D:\\MARIVISION\\H0_workbench\\train_spatial_units.csv\n'
                '    cfg = "C:/Users/admin/SeaClips.zip" # pad(0) and ref_pts with is_tp and best_raw_query\n',
                encoding='utf-8'
            )
            m.write_trace(out,proj)
            trace_txt=(out/'source_decoder_trace.md').read_text(encoding='utf-8')
            self.assertIn('Candidate source locations',trace_txt)
            self.assertIn('<REDACTED_LOCAL_PATH>',trace_txt)
            self.assertIsNone(m.re.search(r'\b[A-Za-z]:[\\/]',trace_txt))
            for bad in ('MARIVISION','train_spatial_units.csv','SeaClips.zip','is_tp','best_raw_query'):
                self.assertNotIn(bad,trace_txt)

    def test_keyboard_interrupt_and_empty_staging_window_roll_back_without_replacing_registry_lists(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            # 1) Simulate async KeyboardInterrupt right after mkdir in 2nd tmp_output before manifest is written
            sub1=root/'case_interrupt_mkdir';sub1.mkdir()
            args1,pins1,rows1,allow1,imgs1,anns1=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(sub1)
            orig_write_man=m.write_invocation_manifest
            calls=[]
            def interrupt_before_second_manifest(folder,files,invocation_token=None,runner_sha256=None):
                calls.append(Path(folder))
                if len(calls)==2:
                    # Neither replace ACTIVE_CREATED_STAGING nor write manifest: simulate async KeyboardInterrupt
                    raise KeyboardInterrupt('Simulated interrupt before second staging manifest')
                return orig_write_man(folder,files,invocation_token=invocation_token,runner_sha256=runner_sha256)
            with patch.object(m,'verify_sources',return_value=pins1), \
                 patch.object(m,'load_whitelist',return_value=(rows1,allow1,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs1,anns1)), \
                 patch.object(m,'write_invocation_manifest',side_effect=interrupt_before_second_manifest):
                with self.assertRaises(KeyboardInterrupt):
                    m.build(args1)
            self.assertFalse((sub1/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((sub1/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

            # 2) Simulate async KeyboardInterrupt immediately after 2nd mkdir() returns (before ACTIVE_CREATED_STAGING.append)
            sub2=root/'case_interrupt_post_mkdir';sub2.mkdir()
            args2,pins2,rows2,allow2,imgs2,anns2=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(sub2)
            orig_mkdir=Path.mkdir
            mkdir_Staging_calls=[]
            def interrupt_right_after_second_staging_mkdir(self_path,*a,**kw):
                res=orig_mkdir(self_path,*a,**kw)
                if str(self_path).endswith('.staging'):
                    mkdir_Staging_calls.append(self_path)
                    if len(mkdir_Staging_calls)==2:
                        raise KeyboardInterrupt('Simulated async interrupt on mkdir return')
                return res
            with patch.object(m,'verify_sources',return_value=pins2), \
                 patch.object(m,'load_whitelist',return_value=(rows2,allow2,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs2,anns2)), \
                 patch.object(Path,'mkdir',autospec=True,side_effect=interrupt_right_after_second_staging_mkdir):
                with self.assertRaises(KeyboardInterrupt):
                    m.build(args2)
            self.assertFalse((sub2/'reviewer_out'/'P2588_identity_build.staging').exists())
            self.assertFalse((sub2/'private_out'/'P2588_sealed_cluster_outcomes.staging').exists())

    def test_zero_intervening_statements_between_mkdir_and_try_in_tmp_output(self):
        import ast,inspect
        tree=ast.parse(inspect.getsource(m.tmp_output))
        fn=tree.body[0]
        mkdir_idx=None
        try_idx=None
        for idx,stmt in enumerate(fn.body):
            src_seg=ast.unparse(stmt)
            if 'tmp.mkdir(' in src_seg:
                mkdir_idx=idx
            elif isinstance(stmt,ast.Try):
                try_idx=idx
                break
        self.assertIsNotNone(mkdir_idx)
        self.assertIsNotNone(try_idx)
        self.assertEqual(try_idx,mkdir_idx+1,'try: must immediately follow tmp.mkdir(...) with zero intervening statements')

    def test_v11_installer_and_rollback_accept_v10_and_rollback_refuses_when_outputs_exist(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if ((here.parent/'APPLY_P2588_ZIP_NATIVE_V11.py').is_file() or (here.parent/'APPLY_P2588_ZIP_NATIVE_V12.py').is_file()) else (here/'OCT15_P2588_ZIP_NATIVE_V11')
        inst_file=next((pkg_root/n for n in ('APPLY_P2588_ZIP_NATIVE_V11.py','APPLY_P2588_ZIP_NATIVE_V12.py') if (pkg_root/n).is_file()),pkg_root/'APPLY_P2588_ZIP_NATIVE_V12.py')
        rb_file=next((pkg_root/n for n in ('ROLLBACK_P2588_V11.py','ROLLBACK_P2588_V12.py') if (pkg_root/n).is_file()),pkg_root/'ROLLBACK_P2588_V12.py')
        if not inst_file.is_file():
            self.skipTest('V11/V12 package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v11',inst_file)
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v11',rb_file)
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v10_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v10_test')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':'P2588_V11_PROVENANCE_PAIR_VERIFY_AND_TRACE_REDACTION_UPGRADE',
                'previous_sources':{'v10':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            priv_root=Path(td)/'private'
            apply_mod.apply(proj,fake_pkg,sealed_root=str(priv_root))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            # Simulate an existing output in private root: rollback must refuse!
            (priv_root/'P2588_sealed_cluster_outcomes').mkdir(parents=True)
            with self.assertRaisesRegex(AssertionError,'Existing P2588 result/staging forbids source rollback'):
                rb_mod.rollback(proj,fake_pkg,sealed_root=str(priv_root))
            (priv_root/'P2588_sealed_cluster_outcomes').rmdir()
            # Simulate an existing reviewer staging dir: rollback must refuse!
            rev_stg=sb/'oct14_p2588_forensic'/'P2588_identity_build.staging'
            rev_stg.mkdir(parents=True)
            with self.assertRaisesRegex(AssertionError,'Existing P2588 result/staging forbids source rollback'):
                rb_mod.rollback(proj,fake_pkg,sealed_root=str(priv_root))
            rev_stg.rmdir()
            # When no outputs or staging exist, rollback restores V10 cleanly
            rb_mod.rollback(proj,fake_pkg,sealed_root=str(priv_root))
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v10_runner')



class V12Stage2ZipNativeTwoTreeAndMergeContracts(unittest.TestCase):
    @staticmethod
    def build_stage1_and_reviews(root):
        args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
        with patch.object(m,'verify_sources',return_value=pins), \
             patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
             patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
            m.build(args)
        build_dir=root/'reviewer_out'/'P2588_identity_build'
        manifest=m.read_csv(build_dir/'identity_qa_manifest.csv')
        review_rows=[]
        for idx,r in enumerate(manifest):
            if idx<210:
                cid=r['provisional_cluster_id']
                vid=int(r['video']);cl=int(r['class'])
                review_rows.append({
                    **r,
                    'verification_status':'VERIFIED',
                    'physical_object_id':f'OBJ_V{vid:02d}_C{cl}_{cid}',
                    'identity_evidence':'IMAGE_QA',
                    'verified_by':'chatgpt_blind_qa_v1',
                })
            else:
                review_rows.append({
                    **r,
                    'verification_status':'UNRESOLVED',
                    'physical_object_id':'',
                    'identity_evidence':'',
                    'verified_by':'chatgpt_blind_qa_v1',
                })
        rev_csv=root/'identity_review_completed.csv'
        m.write_csv(rev_csv,review_rows,list(review_rows[0].keys()))
        # Construct valid candidate merges between same-(video,class) verified clusters with earlier_last_frame < later_first_frame
        by_vc={}
        for r in review_rows:
            if r['verification_status']=='VERIFIED':
                by_vc.setdefault((int(r['video']),int(r['class']),r['class_name']),[]).append(r)
        merge_rows=[]
        for (vid,cl,cname),items in sorted(by_vc.items()):
            items_sorted=sorted(items,key=lambda x:(int(x['first_frame']),int(x['last_frame']),x['provisional_cluster_id']))
            for a,b in zip(items_sorted,items_sorted[1:]):
                f_last=int(a['last_frame']);f_first=int(b['first_frame'])
                if f_last<f_first and (f_first-f_last)%10==0:
                    merge_rows.append({
                        'video':vid,'class':cl,'class_name':cname,
                        'earlier_cluster_id':a['provisional_cluster_id'],
                        'later_cluster_id':b['provisional_cluster_id'],
                        'earlier_last_frame':f_last,'later_first_frame':f_first,
                        'frame_gap':f_first-f_last,
                        'visual_and_kinematic_rationale':'Synthetic split track across frame gap',
                        'merge_confidence':'MEDIUM',
                    })
                    if len(merge_rows)>=6:
                        break
            if len(merge_rows)>=6:
                break
        merges_csv=root/'candidate_cluster_merges.csv'
        m.write_csv(merges_csv,merge_rows,list(m.CANDIDATE_MERGE_COLUMNS))
        args.identity_review=str(rev_csv)
        args.candidate_merges=str(merges_csv)
        args.authorize_stage2_adjudicate=True
        return args,pins,rows,allow_imgs,imgs,anns,manifest,review_rows,merge_rows

    def test_stage2_adjudicate_requires_explicit_director_authorization_in_main_and_adjudicate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns,_,_,_=self.build_stage1_and_reviews(root)
            args.phase='adjudicate'
            args.authorize_stage2_adjudicate=False
            with self.assertRaisesRegex(AssertionError,'Stage 2 human adjudication on HOLD; pass --authorize-stage2-adjudicate'):
                m.adjudicate(args)
            self.assertFalse((root/'reviewer_out'/'P2588_adjudication').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_adjudication_outcomes').exists())
            with patch.object(sys,'argv',['runner','--phase','adjudicate','--output',str(root/'reviewer_out'),'--sealed-root',str(root/'private_out'),'--identity-review',args.identity_review]):
                with self.assertRaisesRegex(AssertionError,'Stage 2 human adjudication on HOLD; pass --authorize-stage2-adjudicate'):
                    m.main()

    def test_review_preflight_is_read_only_and_validates_stage1_pair_and_merges(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns,_,_,_=self.build_stage1_and_reviews(root)
            args.phase='review-preflight'
            args.authorize_stage2_adjudicate=False
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.adjudicate(args)
            self.assertFalse((root/'reviewer_out'/'P2588_adjudication').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_adjudication_outcomes').exists())

    def test_zip_native_stage2_adjudicate_renders_all_40_blind_cases_and_separates_rater_and_private_trees(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns,_,_,merge_rows=self.build_stage1_and_reviews(root)
            args.phase='adjudicate'
            args.authorize_stage2_adjudicate=True
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.adjudicate(args)
            rater_dir=root/'reviewer_out'/'P2588_adjudication'
            priv_dir=root/'private_out'/'P2588_sealed_adjudication_outcomes'
            self.assertTrue(rater_dir.is_dir())
            self.assertTrue(priv_dir.is_dir())
            res=m.verify_published_adjudication_pair(rater_dir,priv_dir,expected_runner_sha=m.current_runner_sha256())
            self.assertTrue(res['both_stage2_trees_paired_and_verified'])
            self.assertEqual(res['blind_unique_object_cases'],40)
            self.assertEqual(res['stage1_blind_panels'],40)
            self.assertEqual(res['stage2_blind_ab_panels'],80)
            # CLI --phase verify-adjudication also succeeds read-only
            with patch.object(sys,'argv',['runner','--phase','verify-adjudication','--output',str(root/'reviewer_out'),'--sealed-root',str(root/'private_out')]):
                m.main()
            # Verify rater package has zero outcome files and zero A/B key files
            self.assertFalse((rater_dir/'SEALED_case_key_DO_NOT_SHARE_WITH_RATERS.csv').exists())
            self.assertFalse((rater_dir/'verified_objects.csv').exists())
            self.assertFalse((rater_dir/'per_object_persistence.csv').exists())
            self.assertFalse((rater_dir/'merged_object_persistence.csv').exists())
            # Verify private package has all sealed key and persistence tables
            self.assertTrue((priv_dir/'SEALED_case_key_DO_NOT_SHARE_WITH_RATERS.csv').is_file())
            self.assertTrue((priv_dir/'verified_objects.csv').is_file())
            self.assertTrue((priv_dir/'per_object_persistence.csv').is_file())
            self.assertTrue((priv_dir/'merged_object_persistence.csv').is_file())
            priv_rep=json.loads((priv_dir/'object_forensic_report.json').read_text(encoding='utf-8'))
            self.assertEqual(priv_rep['human_attested_distinct_object_ids'],210)
            self.assertEqual(priv_rep['n_merge_consolidated_physical_objects'],210-len(merge_rows))
            self.assertEqual(priv_rep['stage1_attestation_modes'],['AI_BLIND_QA_ATTESTED'])

    def test_stage2_two_tree_rollback_cleans_up_both_trees_on_publication_or_manifest_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns,_,_,_=self.build_stage1_and_reviews(root)
            args.phase='adjudicate'
            args.authorize_stage2_adjudicate=True
            orig_pub=m.publish_staged_output
            calls=[]
            def fail_after_second_stage2_rename(tmp,dest,published_tracker=None):
                calls.append(dest)
                orig_pub(tmp,dest,published_tracker=published_tracker)
                if len(calls)==2:
                    raise RuntimeError('Simulated post-rename failure during Stage 2 publication')
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)), \
                 patch.object(m,'publish_staged_output',side_effect=fail_after_second_stage2_rename):
                with self.assertRaisesRegex(RuntimeError,'Simulated post-rename failure during Stage 2 publication'):
                    m.adjudicate(args)
            self.assertFalse((root/'reviewer_out'/'P2588_adjudication').exists())
            self.assertFalse((root/'reviewer_out'/'P2588_adjudication.staging').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_adjudication_outcomes').exists())
            self.assertFalse((root/'private_out'/'P2588_sealed_adjudication_outcomes.staging').exists())
            # Stage 1 outputs must remain completely untouched!
            self.assertTrue((root/'reviewer_out'/'P2588_identity_build').is_dir())
            self.assertTrue((root/'private_out'/'P2588_sealed_cluster_outcomes').is_dir())

    def test_validate_candidate_merges_rejects_cross_video_cross_class_negative_gap_and_frame_overlap(self):
        c1={'provisional_cluster_id':'P0001','video':2,'class':6,'class_name':'SHIP','n_observations':4,
            'first_frame':0,'last_frame':30,'annotation_ids':'1;2;3;4',
            'verification_status':'VERIFIED','physical_object_id':'OBJ_1','identity_evidence':'IMAGE_QA','verified_by':'chatgpt_blind_qa_v1'}
        c2={'provisional_cluster_id':'P0002','video':2,'class':6,'class_name':'SHIP','n_observations':4,
            'first_frame':40,'last_frame':70,'annotation_ids':'5;6;7;8',
            'verification_status':'VERIFIED','physical_object_id':'OBJ_2','identity_evidence':'IMAGE_QA','verified_by':'chatgpt_blind_qa_v1'}
        c3_other_vid={**c2,'provisional_cluster_id':'P0003','video':4,'physical_object_id':'OBJ_3','annotation_ids':'9;10;11;12'}
        manifest=[c1,c2,c3_other_vid]
        verified,_=m.validate_human_decisions(manifest,manifest)
        base_merge={
            'video':2,'class':6,'class_name':'SHIP','earlier_cluster_id':'P0001','later_cluster_id':'P0002',
            'earlier_last_frame':30,'later_first_frame':40,'frame_gap':10,
            'visual_and_kinematic_rationale':'Valid sequential ship fragments','merge_confidence':'MEDIUM'
        }
        v_m,comp_map,m_comps=m.validate_candidate_merges(manifest,verified,[base_merge])
        self.assertEqual(len(v_m),1)
        self.assertEqual(len(m_comps),2)
        self.assertEqual(comp_map['P0001'],comp_map['P0002'])
        # Cross-video merge must fail closed
        with self.assertRaisesRegex(AssertionError,'video mismatch'):
            m.validate_candidate_merges(manifest,verified,[{**base_merge,'later_cluster_id':'P0003'}])
        # Bad frame gap must fail closed
        with self.assertRaisesRegex(AssertionError,'invalid frame_gap'):
            m.validate_candidate_merges(manifest,verified,[{**base_merge,'frame_gap':20}])
        # Invalid confidence must fail closed
        with self.assertRaisesRegex(AssertionError,'Invalid merge_confidence'):
            m.validate_candidate_merges(manifest,verified,[{**base_merge,'merge_confidence':'GUESS'}])

    def test_merge_aware_select_blind_cases_prioritizes_distinct_merge_components_before_backfill(self):
        # Create 7 verified fragments in Group 2 (video=4, class=2) belonging to 2 merge components (M1 has 6 fragments, M2 has 1 fragment)
        clusters=[]
        comp_map={}
        for i in range(1,8):
            cid=f'P{i:04d}'
            oid=f'OBJ_V04_C2_{cid}'
            clusters.append({
                'provisional_cluster_id':cid,'physical_object_id':oid,
                'video':4,'class':2,'class_name':'MARINE_MARKER',
                'n_observations':1,'first_frame':(i-1)*10,'last_frame':(i-1)*10,
                'annotation_ids':str(i),'n_fn':1,'n_tp':0
            })
            comp_map[cid]='MCOMP_A' if i<=6 else 'MCOMP_B'
        chosen=m.select_blind_cases(clusters,seed=m.SEED,k=6,merge_component_by_cid=comp_map)
        self.assertEqual(len(chosen),6)
        chosen_cids={c['provisional_cluster_id'] for c in chosen}
        # Because MCOMP_B is a distinct component in Group 2, Pass 1 must always include P0007 (MCOMP_B) plus 1 from MCOMP_A,
        # and Pass 2 backfills 4 remaining slots from MCOMP_A to reach quota=6!
        self.assertIn('P0007',chosen_cids)

    def test_verify_published_build_pair_accepts_pinned_v11_stage1_build_in_v12(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args,pins,rows,allow_imgs,imgs,anns=V6CompleteOutcomeIndependentQAPanelsContracts.synthetic_build_env(root)
            with patch.object(m,'verify_sources',return_value=pins), \
                 patch.object(m,'load_whitelist',return_value=(rows,allow_imgs,{})), \
                 patch.object(m,'checked_authorized_coco',return_value=(imgs,anns)):
                m.build(args)
            rev_dir=root/'reviewer_out'/'P2588_identity_build'
            priv_dir=root/'private_out'/'P2588_sealed_cluster_outcomes'
            # Rewrite Stage 1 manifests and reports to emulate the Gate F V11 build on the workstation
            for jpath in (rev_dir/'object_forensic_report.json',priv_dir/'SEALED_README.json'):
                data=json.loads(jpath.read_text(encoding='utf-8'))
                data['schema']=m.STAGE1_V11_SCHEMA
                data['runner_sha256']=m.STAGE1_V11_RUNNER_SHA256
                jpath.write_text(json.dumps(data,indent=2),encoding='utf-8')
            for tree in (rev_dir,priv_dir):
                mpath=tree/'SHA256_MANIFEST.json'
                man=json.loads(mpath.read_text(encoding='utf-8'))
                man['schema']=m.STAGE1_V11_SCHEMA
                man['runner_sha256']=m.STAGE1_V11_RUNNER_SHA256
                man['files']={str(p.relative_to(tree)).replace('\\','/'):m.sha(p)
                              for p in sorted(tree.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
                mpath.write_text(json.dumps(man,indent=2),encoding='utf-8')
            res=m.verify_published_build_pair(
                rev_dir,priv_dir,expected_runner_sha=m.current_runner_sha256(),allow_pinned_stage1_v11=True
            )
            self.assertTrue(res['both_trees_paired_and_verified'])
            self.assertEqual(res['schema'],m.STAGE1_V11_SCHEMA)
            self.assertEqual(res['runner_sha256'],m.STAGE1_V11_RUNNER_SHA256)

    def test_v12_installer_preserves_existing_stage1_build_pair_and_refuses_stage2_outputs(self):
        import importlib.util,shutil
        here=Path(__file__).resolve().parent
        pkg_root=here.parent if (here.parent/'APPLY_P2588_ZIP_NATIVE_V12.py').is_file() else (here/'OCT15_P2588_ZIP_NATIVE_V12')
        if not (pkg_root/'APPLY_P2588_ZIP_NATIVE_V12.py').is_file():
            self.skipTest('V12 package root installer not alongside staging directory')
        spec_a=importlib.util.spec_from_file_location('apply_v12',pkg_root/'APPLY_P2588_ZIP_NATIVE_V12.py')
        apply_mod=importlib.util.module_from_spec(spec_a);spec_a.loader.exec_module(apply_mod)
        spec_r=importlib.util.spec_from_file_location('rollback_v12',pkg_root/'ROLLBACK_P2588_V12.py')
        rb_mod=importlib.util.module_from_spec(spec_r);spec_r.loader.exec_module(rb_mod)
        manifest=json.loads((pkg_root/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            proj=Path(td)/'eval_tools';proj.mkdir()
            sb=proj/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
            (sb/'oct13_spatial_matched_control').mkdir(parents=True)
            (sb/'oct13_recoverability_gate').mkdir(parents=True)
            (sb/'oct13_spatial_matched_control'/'train_spatial_units.csv').write_bytes(b'csv')
            (sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json').write_bytes(b'spatial')
            (sb/'oct13_recoverability_gate'/'train_fn_recoverability.json').write_bytes(b'fn')
            (proj/'run_p2588_object_forensics.py').write_bytes(b'v11_runner')
            (proj/'test_run_p2588_object_forensics.py').write_bytes(b'v11_test')
            # Populate a paired Stage 1 build on disk to verify V12 upgrades cleanly over Gate F!
            s1_rev=sb/'oct14_p2588_forensic'/'P2588_identity_build'
            priv_root=Path(td)/'private'
            s1_priv=priv_root/'P2588_sealed_cluster_outcomes'
            s1_rev.mkdir(parents=True);s1_priv.mkdir(parents=True)
            (s1_rev/'SHA256_MANIFEST.json').write_text(json.dumps({'invocation_token':'tok_gate_f'}),encoding='utf-8')
            (s1_priv/'SHA256_MANIFEST.json').write_text(json.dumps({'invocation_token':'tok_gate_f'}),encoding='utf-8')
            fake_pkg=Path(td)/'pkg';(fake_pkg/'staging').mkdir(parents=True)
            for name in ('run_p2588_object_forensics.py','test_run_p2588_object_forensics.py'):
                shutil.copy2(pkg_root/'staging'/name,fake_pkg/'staging'/name)
            fake_manifest={
                'schema':'P2588_V12_STAGE2_ZIP_NATIVE_TWO_TREE_AND_MERGE_UPGRADE',
                'previous_sources':{'v11':{
                    'run_p2588_object_forensics.py':apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                    'test_run_p2588_object_forensics.py':apply_mod.sha(proj/'test_run_p2588_object_forensics.py'),
                }},
                'new_sources':manifest['new_sources'],
                'pinned_upstream':{
                    'train_spatial_units.csv':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_units.csv'),
                    'train_spatial_diagnostic.json':apply_mod.sha(sb/'oct13_spatial_matched_control'/'train_spatial_diagnostic.json'),
                    'train_fn_recoverability.json':apply_mod.sha(sb/'oct13_recoverability_gate'/'train_fn_recoverability.json'),
                }
            }
            (fake_pkg/'PATCH_MANIFEST.json').write_text(json.dumps(fake_manifest),encoding='utf-8')
            apply_mod.apply(proj,fake_pkg,sealed_root=str(priv_root))
            self.assertEqual(apply_mod.sha(proj/'run_p2588_object_forensics.py'),
                             manifest['new_sources']['run_p2588_object_forensics.py'])
            # Existing Stage 2 output must block rollback
            s2_priv=priv_root/'P2588_sealed_adjudication_outcomes'
            s2_priv.mkdir(parents=True)
            with self.assertRaisesRegex(AssertionError,'Existing P2588 result/staging forbids source rollback'):
                rb_mod.rollback(proj,fake_pkg,sealed_root=str(priv_root))
            s2_priv.rmdir()
            # With no Stage 2 outputs, rollback to V11 succeeds while preserving Stage 1 pair
            rb_mod.rollback(proj,fake_pkg,sealed_root=str(priv_root))
            self.assertEqual((proj/'run_p2588_object_forensics.py').read_bytes(),b'v11_runner')
