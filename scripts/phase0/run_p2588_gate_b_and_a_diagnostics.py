"""P-2588 Next-Gate Read-Only Diagnostics: Gate B (271-FN Taxonomy & Recoverability Ceiling) + Gate A (H_S5 Input Transform & O.r Coordinate Frame).

Strictly read-only, TRAIN-only, CPU-only diagnostic runner.
- Does NOT import or run any detector forward pass or training loop.
- Does NOT touch VAL or TEST partitions.
- Maintains no_architecture_authorization = True.
- Joins all 1,025 TRAIN GT observations (754 TP + 271 FN) across all 240 Stage 1 clusters
  (207 VERIFIED [733 obs: 584 TP, 149 FN] + 33 UNRESOLVED [292 obs: 170 TP, 122 FN]),
  183 MEDIUM-merged physical objects (and 170 MEDIUM+LOW sensitivity objects),
  train_spatial_units.csv, train_fn_recoverability.json, and train_spatial_diagnostic.json.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_VIDEOS = (2, 3, 4, 5, 7, 8, 13, 14, 16, 17, 18, 19, 20, 24, 25, 31, 35)
FORBIDDEN_VIDEOS = frozenset((6, 29, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 1, 9, 10, 22, 26, 27, 28, 32, 34, 11, 12, 15, 21, 23, 30, 33, 39, 46))
CLASS_NAMES = ('ANIMAL', 'BOAT', 'MARINE_MARKER', 'OBJECT', 'LEISURE_VEHICLE', 'SAILING_VESSEL', 'SHIP')

TRAIN_COCO_SHA = 'edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35'
SPATIAL_CSV_SHA = '941110163dede4fb1aa14c8b7e1a0e8d48cecde061ab33c6adafdaab78a32ad3'
SPATIAL_JSON_SHA = 'fb2b486cea80bc4619c9e9707bf1806ced1751bbc5a5516eebc6f8dcd382a5a5'
FN_JSON_SHA = '358c8d277a3ef354477c0b7cea7afd3fffd6c4a6960281f5d8ad063aec67c5c3'


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p: Path) -> list[dict[str, str]]:
    with Path(p).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with Path(p).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def compute_oracle_iou_from_offsets(dx_over_w: float, dy_over_h: float, log_pw_over_w: float, log_ph_over_h: float) -> float:
    """Exact IoU between unit GT box [-0.5, 0.5]^2 and ORACLE box reconstructed from relative offsets."""
    pw = math.exp(log_pw_over_w)
    ph = math.exp(log_ph_over_h)
    gx0, gy0, gx1, gy1 = -0.5, -0.5, 0.5, 0.5
    px0, py0 = dx_over_w - 0.5 * pw, dy_over_h - 0.5 * ph
    px1, py1 = dx_over_w + 0.5 * pw, dy_over_h + 0.5 * ph
    ix = max(0.0, min(gx1, px1) - max(gx0, px0))
    iy = max(0.0, min(gy1, py1) - max(gy0, py0))
    inter = ix * iy
    union = 1.0 + (pw * ph) - inter
    return inter / max(1e-12, union)


def build_merge_components(clusters: list[dict], merges: list[dict], allowed_tiers: set[str], allow_unresolved: bool = False) -> dict[str, str]:
    status_by_cid = {c['provisional_cluster_id']: c['verification_status'].strip().upper() for c in clusters}
    valid_cids = {cid for cid, st in status_by_cid.items() if allow_unresolved or st == 'VERIFIED'}
    parent = {cid: cid for cid in valid_cids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in merges:
        tier = m['merge_confidence'].strip().upper()
        if tier not in allowed_tiers:
            continue
        a, b = m['earlier_cluster_id'].strip(), m['later_cluster_id'].strip()
        if a in valid_cids and b in valid_cids:
            ra, rb = find(a), find(b)
            if ra != rb:
                if int(ra[1:]) <= int(rb[1:]):
                    parent[rb] = ra
                else:
                    parent[ra] = rb
    comp_members = defaultdict(list)
    for cid in sorted(valid_cids):
        comp_members[find(cid)].append(cid)
    comp_id_by_cid = {}
    by_cid = {c['provisional_cluster_id']: c for c in clusters}
    for root_cid, members in comp_members.items():
        c0 = by_cid[members[0]]
        comp_id = f"MCOMP_V{int(c0['video']):02d}_C{int(c0['class'])}_{members[0]}"
        for cid in members:
            comp_id_by_cid[cid] = comp_id
    return comp_id_by_cid


def run_diagnostics(args: argparse.Namespace) -> dict:
    project = Path(args.project).resolve()
    spatial_csv = project / 'bfcef_f_s0/run15a_final_candidate_sandbox/oct13_spatial_matched_control/train_spatial_units.csv'
    spatial_json = project / 'bfcef_f_s0/run15a_final_candidate_sandbox/oct13_spatial_matched_control/train_spatial_diagnostic.json'
    fn_json = project / 'bfcef_f_s0/run15a_final_candidate_sandbox/oct13_recoverability_gate/train_fn_recoverability.json'
    coco_path = Path(args.coco_train).resolve()
    identity_review_path = Path(args.identity_review).resolve()
    merges_path = Path(args.candidate_merges).resolve()
    stage1_build_dir = project / 'bfcef_f_s0/run15a_final_candidate_sandbox/oct14_p2588_forensic/P2588_identity_build'

    for label, path, expected_sha in (
        ('coco_train', coco_path, TRAIN_COCO_SHA),
        ('train_spatial_units.csv', spatial_csv, SPATIAL_CSV_SHA),
        ('train_spatial_diagnostic.json', spatial_json, SPATIAL_JSON_SHA),
        ('train_fn_recoverability.json', fn_json, FN_JSON_SHA),
    ):
        assert path.is_file(), f"Missing pinned input {label}: {path}"
        got = sha256_file(path)
        assert got == expected_sha, f"SHA-256 mismatch for {label}: {got} != {expected_sha}"

    unit_rows = read_csv(spatial_csv)
    assert len(unit_rows) == 1025, f"Expected 1025 rows in train_spatial_units.csv, got {len(unit_rows)}"
    spatial_diag = json.loads(spatial_json.read_text(encoding='utf-8'))
    fn_rec = json.loads(fn_json.read_text(encoding='utf-8'))
    assert len(fn_rec.get('fn_events', [])) == 271, "Expected 271 fn_events in train_fn_recoverability.json"

    review_rows = read_csv(identity_review_path)
    merge_rows = read_csv(merges_path)
    assert len(review_rows) == 240, f"Expected 240 Stage 1 clusters, got {len(review_rows)}"

    spacing_path = stage1_build_dir / 'gt_neighbor_spacing.csv'
    spacing_by_aid = {}
    if spacing_path.is_file():
        for r in read_csv(spacing_path):
            spacing_by_aid[int(r['annotation_id'])] = r

    # Index fn_events by (video, frame, image_id, gt_index)
    fn_event_by_key = {}
    fn_event_sample_keys = []
    for ev in fn_rec['fn_events']:
        k = (int(ev['video']), int(ev['frame']), int(ev['image_id']), int(ev['g']))
        fn_event_by_key[k] = ev
        if len(fn_event_sample_keys) < 1:
            fn_event_sample_keys = sorted(ev.keys())

    # Map annotation_id -> cluster info
    cluster_by_aid = {}
    cluster_by_cid = {}
    for c in review_rows:
        cid = c['provisional_cluster_id'].strip()
        cluster_by_cid[cid] = c
        for aid_str in c['annotation_ids'].split(';'):
            cluster_by_aid[int(aid_str)] = c

    assert len(cluster_by_aid) == 1025, f"Expected 1025 annotations across 240 clusters, got {len(cluster_by_aid)}"

    # Build merge components under MEDIUM (verified only), MEDIUM+LOW (verified only), and ALL (verified+unresolved)
    comp_med_ver = build_merge_components(review_rows, merge_rows, {'HIGH', 'MEDIUM'}, allow_unresolved=False)
    comp_all_ver = build_merge_components(review_rows, merge_rows, {'HIGH', 'MEDIUM', 'LOW'}, allow_unresolved=False)
    comp_all_240 = build_merge_components(review_rows, merge_rows, {'HIGH', 'MEDIUM', 'LOW'}, allow_unresolved=True)

    # Enrich all 1,025 GT observations
    enriched_units = []
    stratum_summary = defaultdict(lambda: Counter())
    video_summary = defaultdict(lambda: Counter())
    fn_taxonomy_counts = Counter()
    fn_iou_buckets = Counter()

    for r in unit_rows:
        aid = int(r['annotation_id'])
        vid = int(r['video'])
        cl = int(r['class'])
        frame = int(r['frame'])
        iid = int(r['image_id'])
        g_idx = int(r['gt_index'])
        is_tp = int(r['is_tp'])
        assert vid in TRAIN_VIDEOS and vid not in FORBIDDEN_VIDEOS

        c = cluster_by_aid[aid]
        cid = c['provisional_cluster_id'].strip()
        v_status = c['verification_status'].strip().upper()

        dx = float(r['signed_center_x_over_gt_w'])
        dy = float(r['signed_center_y_over_gt_h'])
        lpw = float(r['log_pred_w_over_gt_w'])
        lph = float(r['log_pred_h_over_gt_h'])
        oracle_iou_recomputed = compute_oracle_iou_from_offsets(dx, dy, lpw, lph)
        center_disp_in_gt_diag = math.hypot(dx, dy)
        area_ratio_pred_over_gt = math.exp(lpw + lph)
        gt_area = float(r['gt_area_source_px2'])

        sp = spacing_by_aid.get(aid, {})
        cy_px = float(sp['cy_source_px']) if sp.get('cy_source_px') not in (None, '') else None
        nn_geomean = float(sp['nearest_anyclass_gt_center_in_gt_geomean']) if sp.get('nearest_anyclass_gt_center_in_gt_geomean') not in (None, '') else None

        ev = fn_event_by_key.get((vid, frame, iid, g_idx), {}) if is_tp == 0 else {}
        # Extract any IoU, score, or rank fields present in train_fn_recoverability.json or train_spatial_units.csv
        ev_iou = ev.get('best_iou', ev.get('iou', ev.get('max_iou', r.get('best_raw_iou', None))))
        ev_score = ev.get('best_score', ev.get('score', ev.get('target_score', ev.get('prob', r.get('best_raw_score', None)))))
        ev_rank = ev.get('rank', ev.get('score_rank', ev.get('query_rank', None)))
        ev_cat = ev.get('category', ev.get('recoverability', ev.get('fn_type', ev.get('bucket', None))))

        oracle_iou = float(ev_iou) if ev_iou not in (None, '') else oracle_iou_recomputed

        # Stratum & mechanistic classification for FNs
        if is_tp == 0:
            if oracle_iou >= 0.50:
                iou_bucket = 'IOU_GE_0.50_SCORE_OR_ASSIGNMENT_MISS'
            elif oracle_iou >= 0.30:
                iou_bucket = 'IOU_0.30_TO_0.50_BOUNDARY_LOCALIZATION'
            elif oracle_iou >= 0.10:
                iou_bucket = 'IOU_0.10_TO_0.30_POOR_EXTENT_OR_OFFSET'
            elif oracle_iou > 0.0:
                iou_bucket = 'IOU_0.00_TO_0.10_NEAR_ZERO_OVERLAP'
            else:
                iou_bucket = 'IOU_EQ_0.00_DISJOINT_OR_HIJACKED_QUERY'
            fn_iou_buckets[(v_status, iou_bucket)] += 1

            # Primary forensic mechanism bucket (aligned with Claude's Gate B recommendation)
            if vid == 18 and cl == 0:
                mech = 'M1_V18_C0_QUAY_FACADE_ANNOTATION_ARTIFACT'
            elif vid == 18 and cl == 2:
                if area_ratio_pred_over_gt >= 4.0 or oracle_iou < 0.05:
                    mech = 'M2A_V18_C2_ADJACENT_VESSEL_HIJACK_OR_DISJOINT'
                else:
                    mech = 'M2B_V18_C2_HARBOR_SPECK_OFFSET_OR_LOW_SCORE'
            elif gt_area <= 16.0 or (vid == 4 and cl == 2):
                mech = 'M3A_SUBRESOLUTION_SPECK_LE_16PX2'
            elif vid == 24 and cl == 3:
                mech = 'M3B_V24_C3_SMALL_WATER_OR_HORIZON_STREAK'
            else:
                mech = 'M4_OTHER_STRATA_FN'
            fn_taxonomy_counts[(v_status, mech)] += 1
        else:
            iou_bucket = 'TP_DETECTED'
            mech = 'TP_DETECTED'

        key_vc = f"v{vid:02d}_c{cl}_{CLASS_NAMES[cl]}"
        stratum_summary[key_vc]['total_obs'] += 1
        stratum_summary[key_vc]['tp' if is_tp else 'fn'] += 1
        stratum_summary[key_vc][f"{v_status.lower()}_obs"] += 1
        stratum_summary[key_vc][f"{v_status.lower()}_{'tp' if is_tp else 'fn'}"] += 1

        video_summary[vid]['total_obs'] += 1
        video_summary[vid]['tp' if is_tp else 'fn'] += 1
        video_summary[vid][f"{v_status.lower()}_obs"] += 1
        video_summary[vid][f"{v_status.lower()}_{'tp' if is_tp else 'fn'}"] += 1

        enriched_units.append({
            'annotation_id': aid,
            'video': vid,
            'class': cl,
            'class_name': CLASS_NAMES[cl],
            'frame': frame,
            'image_id': iid,
            'provisional_cluster_id': cid,
            'stage1_status': v_status,
            'merge_comp_medium_verified': comp_med_ver.get(cid, f"UNRES_{cid}"),
            'merge_comp_all37_verified': comp_all_ver.get(cid, f"UNRES_{cid}"),
            'merge_comp_all45_including_unresolved': comp_all_240.get(cid, cid),
            'is_tp': is_tp,
            'gt_area_source_px2': round(gt_area, 2),
            'cy_source_px': round(cy_px, 2) if cy_px is not None else '',
            'nearest_gt_geomean': round(nn_geomean, 3) if nn_geomean is not None else '',
            'best_raw_query': r['best_raw_query'],
            'oracle_iou_recomputed': round(oracle_iou_recomputed, 4),
            'oracle_iou_receipt': round(float(ev_iou), 4) if ev_iou not in (None, '') else '',
            'oracle_score_receipt': ev_score if ev_score not in (None, '') else '',
            'oracle_rank_receipt': ev_rank if ev_rank not in (None, '') else '',
            'fn_receipt_category': ev_cat if ev_cat not in (None, '') else '',
            'pred_over_gt_area_ratio': round(area_ratio_pred_over_gt, 3),
            'center_disp_in_gt_box_units': round(center_disp_in_gt_diag, 3),
            'fn_iou_bucket': iou_bucket,
            'forensic_mechanism': mech,
        })

    # Compute persistence across all 4 consolidation views:
    # (1) 207 VERIFIED unmerged, (2) 183 VERIFIED MEDIUM-merged, (3) 170 VERIFIED MEDIUM+LOW-merged, (4) All 240 clusters (VERIFIED + UNRESOLVED)
    def summarize_persistence(unit_list: list[dict], group_key: str) -> dict:
        by_grp = defaultdict(list)
        for u in unit_list:
            by_grp[u[group_key]].append(u)
        n_groups = len(by_grp)
        pers_80 = [g for g, obs in by_grp.items() if sum(1 - x['is_tp'] for x in obs) / len(obs) >= 0.8]
        pers_80_ge3 = [g for g in pers_80 if len(by_grp[g]) >= 3]
        pers_80_2 = [g for g in pers_80 if len(by_grp[g]) == 2]
        pers_80_1 = [g for g in pers_80 if len(by_grp[g]) == 1]
        fn_in_pers_80 = sum(sum(1 - x['is_tp'] for x in by_grp[g]) for g in pers_80)
        fn_in_pers_80_ge3 = sum(sum(1 - x['is_tp'] for x in by_grp[g]) for g in pers_80_ge3)
        total_fn = sum(1 - x['is_tp'] for x in unit_list)
        return {
            'n_distinct_units': n_groups,
            'n_observations': len(unit_list),
            'n_tp': sum(x['is_tp'] for x in unit_list),
            'n_fn': total_fn,
            'n_persistent_ge80pct': len(pers_80),
            'n_persistent_ge80pct_multiframe_ge3': len(pers_80_ge3),
            'n_persistent_ge80pct_two_frame': len(pers_80_2),
            'n_persistent_ge80pct_singleton': len(pers_80_1),
            'fn_in_persistent_ge80pct': fn_in_pers_80,
            'fn_share_in_persistent_ge80pct': round(fn_in_pers_80 / max(1, total_fn), 4),
            'fn_in_persistent_ge80pct_multiframe_ge3': fn_in_pers_80_ge3,
            'fn_share_in_persistent_ge80pct_multiframe_ge3': round(fn_in_pers_80_ge3 / max(1, total_fn), 4),
        }

    verified_units = [u for u in enriched_units if u['stage1_status'] == 'VERIFIED']
    unresolved_units = [u for u in enriched_units if u['stage1_status'] != 'VERIFIED']

    persistence_sensitivity = {
        'verified_unmerged_207': summarize_persistence(verified_units, 'provisional_cluster_id'),
        'verified_medium_merged_183': summarize_persistence(verified_units, 'merge_comp_medium_verified'),
        'verified_medium_plus_low_merged_170': summarize_persistence(verified_units, 'merge_comp_all37_verified'),
        'unresolved_unmerged_33': summarize_persistence(unresolved_units, 'provisional_cluster_id'),
        'all_240_unmerged_clusters': summarize_persistence(enriched_units, 'provisional_cluster_id'),
        'all_195_merged_including_unresolved': summarize_persistence(enriched_units, 'merge_comp_all45_including_unresolved'),
    }

    # Gate A: Read-only static inspection of H_S5 (input transform, resize, stride, O.r, ref_pts)
    script_names = (
        'train_bfcef_f_s0_final_experts.py',
        'run15a_final_windows.py',
        'run15a_final_mechanism.py',
        'run15a_final_supervision.py',
        'run15a_internal_rebuild_PATCH03.py',
    )
    pat_hs5 = re.compile(
        r'resize|size|max_size|min_size|pad|normalize|mean|std|stride|reference_points|ref_pts|bbox_embed|\bO\.r\b|input_shape|img_shape|scale_factor',
        re.I,
    )
    gate_a_traces = {}
    for sname in script_names:
        sp = project / sname
        if not sp.is_file():
            gate_a_traces[sname] = {'present': False}
            continue
        lines = sp.read_text(encoding='utf-8', errors='replace').splitlines()
        hits = []
        for idx, line in enumerate(lines, 1):
            if pat_hs5.search(line):
                hits.append({'line': idx, 'code': line.strip()[:180]})
        gate_a_traces[sname] = {
            'present': True,
            'sha256': sha256_file(sp),
            'n_matching_lines': len(hits),
            'matching_lines_sample': hits[:60],
        }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / 'P2588_gate_b_all_1025_units_and_271_fn_taxonomy.csv',
        enriched_units,
        list(enriched_units[0].keys()),
    )

    summary_payload = {
        'schema': 'P2588_GATE_B_AND_GATE_A_READONLY_DIAGNOSTIC_V1',
        'no_architecture_authorization': True,
        'train_only': True,
        'unit_csv_columns': list(unit_rows[0].keys()),
        'fn_recoverability_top_keys': sorted(fn_rec.keys()),
        'fn_recoverability_event_keys': fn_event_sample_keys,
        'spatial_diagnostic_top_keys': sorted(spatial_diag.keys()),
        'spatial_diagnostic_summary': {k: v for k, v in spatial_diag.items() if not isinstance(v, list)},
        'fn_recoverability_summary': {k: v for k, v in fn_rec.items() if k != 'fn_events'},
        'persistence_sensitivity': persistence_sensitivity,
        'fn_iou_buckets_by_stage1_status': {f"{st}__{b}": cnt for (st, b), cnt in sorted(fn_iou_buckets.items())},
        'fn_mechanism_taxonomy_by_stage1_status': {f"{st}__{m}": cnt for (st, m), cnt in sorted(fn_taxonomy_counts.items())},
        'per_stratum_tp_fn_verified_vs_unresolved': {k: dict(v) for k, v in sorted(stratum_summary.items())},
        'per_video_tp_fn_verified_vs_unresolved': {str(k): dict(v) for k, v in sorted(video_summary.items())},
        'gate_a_hs5_static_source_trace': gate_a_traces,
    }
    (out_dir / 'P2588_gate_b_and_a_summary.json').write_text(
        json.dumps(summary_payload, indent=2), encoding='utf-8'
    )
    print('[P2588 GATE B + GATE A READ-ONLY DIAGNOSTIC COMPLETE]')
    print(json.dumps({
        'persistence_sensitivity': persistence_sensitivity,
        'fn_iou_buckets_by_stage1_status': summary_payload['fn_iou_buckets_by_stage1_status'],
        'fn_mechanism_taxonomy_by_stage1_status': summary_payload['fn_mechanism_taxonomy_by_stage1_status'],
        'per_stratum_tp_fn_verified_vs_unresolved': summary_payload['per_stratum_tp_fn_verified_vs_unresolved'],
    }, indent=2))
    return summary_payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default=r'D:\MARIVISION\H0_workbench\eval_tools')
    ap.add_argument('--coco-train', default=r'D:\MARIVISION\seaclips-train.json')
    ap.add_argument('--identity-review', default=r'stage1_review_v2_1\identity_review_completed.csv')
    ap.add_argument('--candidate-merges', default=r'stage1_review_v2_1\candidate_cluster_merges.csv')
    ap.add_argument('--out-dir', default=r'D:\MARIVISION\P2588_PRIVATE_OUTCOMES\P2588_gate_b_and_a_diagnostics')
    args = ap.parse_args()
    run_diagnostics(args)


if __name__ == '__main__':
    main()