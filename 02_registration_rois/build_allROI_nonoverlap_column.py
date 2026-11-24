#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build all-ROI non-overlap column masks in functional space.

For each subject:
  1. Load HCP-MMP atlas (func space) + layer file (func space, from Script 3)
  2. For each target ROI: extract voxels matching LH + RH atlas labels
  3. Non-overlap: each GM voxel is assigned to at most one ROI (first match)
  4. Dilate `--dilate` steps into unassigned GM voxels (no overlap created)
  5. Intersect with each cortical layer → per-layer ROI labels
  6. Save label volumes and ROI name list

Output per subject  (OUT_BASE/<tag>/sub-XX/):
  allroi_label.nii.gz        -- combined non-overlap ROI label (1-indexed, int16)
  layer1_roi_label.nii.gz    -- layer-1 ROI labels (deep / WM side)
  layer2_roi_label.nii.gz    -- layer-2 ROI labels (middle)
  layer3_roi_label.nii.gz    -- layer-3 ROI labels (shallow / pial side)
  roi_names.npy              -- 1-D string array, roi_names[i-1] = label i

Usage:
  python build_allROI_nonoverlap_column_masks.py --subject 01
  python build_allROI_nonoverlap_column_masks.py --parallel 4 --force
  python build_allROI_nonoverlap_column_masks.py --dilate 2 --force
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from scipy import ndimage

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh

# ── global paths ─────────────────────────────────────────────────────────────
ROOT           = Path(PROJECT_ROOT + "/derivatives_0331")
ATLAS_FUNC_DIR = Path(PROJECT_ROOT + "/Atlas_HCPMMP/func_space_MPRAGEisedAffine")
LAYER_BASE     = ROOT / "Step1_Anat2Func" / "layers2func_MPRAGEisedAffine"
OUT_BASE       = ROOT / "Step2_ROI_LayerMasks_AllROI_ColumnNonOverlap"

SUBJECTS = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","16"]

# ── HCP-MMP label look-up table ───────────────────────────────────────────────
# LH labels 1002-1181 (1121 missing); RH = LH + 1000
# Source: LUT_HCPMMP1.txt in Atlas_HCPMMP
HCP_MMP_LH_LABELS = {
    "V1":1002,"MST":1003,"V6":1004,"V2":1005,"V3":1006,
    "V4":1007,"V8":1008,"4":1009,"3b":1010,"FEF":1011,
    "PEF":1012,"55b":1013,"V3A":1014,"RSC":1015,"POS2":1016,
    "V7":1017,"IPS1":1018,"FFC":1019,"V3B":1020,"LO1":1021,
    "LO2":1022,"PIT":1023,"MT":1024,"A1":1025,"PSL":1026,
    "SFL":1027,"PCV":1028,"STV":1029,"7Pm":1030,"7m":1031,
    "POS1":1032,"23d":1033,"v23ab":1034,"d23ab":1035,"31pv":1036,
    "5m":1037,"5mv":1038,"23c":1039,"5L":1040,"24dd":1041,
    "24dv":1042,"7AL":1043,"SCEF":1044,"6ma":1045,"7Am":1046,
    "7PL":1047,"7PC":1048,"LIPv":1049,"VIP":1050,"MIP":1051,
    "1":1052,"2":1053,"3a":1054,"6d":1055,"6mp":1056,
    "6v":1057,"p24pr":1058,"33pr":1059,"a24pr":1060,"p32pr":1061,
    "a24":1062,"d32":1063,"8BM":1064,"p32":1065,"10r":1066,
    "47m":1067,"8Av":1068,"8Ad":1069,"9m":1070,"8BL":1071,
    "9p":1072,"10d":1073,"8C":1074,"44":1075,"45":1076,
    "47l":1077,"a47r":1078,"6r":1079,"IFJa":1080,"IFJp":1081,
    "IFSp":1082,"IFSa":1083,"p9-46v":1084,"46":1085,"a9-46v":1086,
    "9-46d":1087,"9a":1088,"10v":1089,"a10p":1090,"10pp":1091,
    "11l":1092,"13l":1093,"OFC":1094,"47s":1095,"LIPd":1096,
    "6a":1097,"i6-8":1098,"s6-8":1099,"43":1100,"OP4":1101,
    "OP1":1102,"OP2-3":1103,"52":1104,"RI":1105,"PFcm":1106,
    "PoI2":1107,"TA2":1108,"FOP4":1109,"MI":1110,"Pir":1111,
    "AVI":1112,"AAIC":1113,"FOP1":1114,"FOP3":1115,"FOP2":1116,
    "PFt":1117,"AIP":1118,"EC":1119,"PreS":1120,
    # 1121 does not exist in this atlas version
    "ProS":1122,"PeEc":1123,"STGa":1124,"PBelt":1125,
    "A5":1126,"PHA1":1127,"PHA3":1128,"STSda":1129,"STSdp":1130,
    "STSvp":1131,"TGd":1132,"TE1a":1133,"TE1p":1134,"TE2a":1135,
    "TF":1136,"TE2p":1137,"PHT":1138,"PH":1139,"TPOJ1":1140,
    "TPOJ2":1141,"TPOJ3":1142,"DVT":1143,"PGp":1144,"IP2":1145,
    "IP1":1146,"IP0":1147,"PFop":1148,"PF":1149,"PFm":1150,
    "PGi":1151,"PGs":1152,"V6A":1153,"VMV1":1154,"VMV3":1155,
    "PHA2":1156,"V4t":1157,"FST":1158,"V3CD":1159,"LO3":1160,
    "VMV2":1161,"31pd":1162,"31a":1163,"VVC":1164,"25":1165,
    "s32":1166,"pOFC":1167,"PoI1":1168,"Ig":1169,"FOP5":1170,
    "p10p":1171,"p47r":1172,"TGv":1173,"MBelt":1174,"LBelt":1175,
    "A4":1176,"STSva":1177,"TE1m":1178,"PI":1179,"a32pr":1180,
    "p24":1181,
}
HCP_MMP_RH_LABELS = {k: v + 1000 for k, v in HCP_MMP_LH_LABELS.items()}

# ── target ROI sets ───────────────────────────────────────────────────────────
OCCIPITAL_ROIS = {
    "V1","V2","V3","V4","V3A","MT","MST","V6","V6A","V7",
    "LO1","LO2","LO3","V4t","FST","V3CD",
    "FFC","VVC","VMV1","VMV2","VMV3","PCV",
}
PPC_ROIS = {
    "IPS1","LIPv","LIPd","VIP","AIP",
    "7PC","7Am","7AL","7PL","7Pm","7m",
    "POS1","POS2","RSC","ProS","DVT",
    "31pd","31a","31pv","v23ab","d23ab","23d","23c",
    "TPOJ1","TPOJ2","TPOJ3","STV",
    "PHA1","PHA2","PHA3",
}
PFC_ROIS = {
    "3a","3b","4",
    "6d","6v","6mp","6r","FEF","PEF","55b",
    "SFL","8BM","8Ad","8C",
    "p9-46v","a9-46v",
    "p24","a24pr",
}
VALID_REGIONS = OCCIPITAL_ROIS | PPC_ROIS | PFC_ROIS


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Build all-ROI non-overlap column masks in func space.")
    p.add_argument("--subject",      default=None,
                   help="Single subject ID (e.g. 01). Omit to run all.")
    p.add_argument("--ld",           default="1000",
                   help="Mesh LD used in layer segmentation (default 1000).")
    p.add_argument("--nr-layers",    default=3, type=int,
                   help="Number of cortical layers (default 3).")
    p.add_argument("--layer-method", default="equivol",
                   help="Layer method (default equivol).")
    p.add_argument("--dilate",       default=1, type=int,
                   help="Dilation steps into unassigned GM (default 1).")
    p.add_argument("--connectivity", default=1, type=int, choices=[1,2,3],
                   help="Morphological connectivity 1/2/3 (default 1=face).")
    p.add_argument("--roi-set",      default="valid_regions",
                   choices=["valid_regions","all180"],
                   help="ROI set to use (default valid_regions).")
    p.add_argument("--parallel",     default=1, type=int,
                   help="Number of parallel workers (default 1).")
    p.add_argument("--force",        action="store_true",
                   help="Overwrite existing output.")
    return p.parse_args()


def tag_from_args(args):
    return "ld%s_%dlayers_%s" % (args.ld, args.nr_layers, args.layer_method)


# ── ROI list builder ──────────────────────────────────────────────────────────
def get_target_rois(args):
    """
    Returns (target_names, label_info_list) for the requested ROI set.
    Uses HCP_MMP_LH_LABELS for direct name→label lookup.
    roi_index is 1-based and corresponds to the label value in the output files.
    """
    # Use HCP_MMP_LH_LABELS key order as the canonical ROI ordering.
    # This replaces the old dependency on roi_names.npy from previous runs.
    all_names = list(HCP_MMP_LH_LABELS.keys())   # canonical order, 179 entries

    if args.roi_set == "all180":
        candidates = all_names
    else:
        candidates = [n for n in all_names if n in VALID_REGIONS]

    target_names = []
    label_info   = []
    for name in candidates:
        lh = HCP_MMP_LH_LABELS.get(name, -1)
        rh = HCP_MMP_RH_LABELS.get(name, -1)
        if lh == -1:
            print("[WARN] '%s' not in label table — skip." % name)
            continue
        target_names.append(name)
        label_info.append({
            "roi_index": len(label_info) + 1,   # 1-based
            "roi_name":  name,
            "lh_label":  lh,
            "rh_label":  rh,
        })

    return target_names, label_info


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_img(path):
    img  = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return img, data


def save_nifti(data, ref_img, out_path, dtype=np.int16):
    out = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    out.header.set_data_dtype(dtype)
    nib.save(out, str(out_path))


# ── core mask builder ─────────────────────────────────────────────────────────
def build_roi_label(atlas_data, gm_mask, label_info, dilate, connectivity):
    """
    Assign each GM voxel to at most one ROI (non-overlapping).
    Order = label_info order; first match wins.
    Then dilate `dilate` times into unassigned GM voxels.

    Returns int16 array: value = roi_index (1-based) or 0 (unassigned).
    """
    struct = ndimage.generate_binary_structure(3, connectivity)
    roi_label = np.zeros(atlas_data.shape, dtype=np.int16)

    # ── initial assignment ────────────────────────────────────────────────────
    for info in label_info:
        idx = info["roi_index"]
        lh  = info["lh_label"]
        rh  = info["rh_label"]
        roi_vox = ((atlas_data == lh) | (atlas_data == rh)) & gm_mask
        # non-overlap: only write where not yet claimed
        roi_label[roi_vox & (roi_label == 0)] = idx

    # ── dilation into unassigned GM ───────────────────────────────────────────
    for _ in range(dilate):
        unassigned = gm_mask & (roi_label == 0)
        if not np.any(unassigned):
            break
        new = roi_label.copy()
        # propagate each label separately to avoid race conditions
        for info in label_info:
            idx  = info["roi_index"]
            mask = roi_label == idx
            if not np.any(mask):
                continue
            dilated   = ndimage.binary_dilation(mask, structure=struct)
            new_voxels = dilated & unassigned & (new == 0)
            new[new_voxels] = idx
        roi_label = new

    return roi_label


# ── per-subject processing ────────────────────────────────────────────────────
def process_subject(sub, args_dict):
    """Worker function (called directly or via ProcessPoolExecutor)."""
    import argparse as _ap
    args = _ap.Namespace(**args_dict)

    tag      = tag_from_args(args)
    out_dir  = OUT_BASE / tag / ("sub-%s" % sub)
    sentinel = out_dir / "roi_names.npy"

    if sentinel.exists() and not args.force:
        print("[SKIP] sub-%s — output exists (use --force to redo)" % sub)
        return sub, True

    # ── paths ─────────────────────────────────────────────────────────────────
    atlas_path  = ATLAS_FUNC_DIR / ("sub-%s_HCPMMP1_in_func.nii.gz" % sub)
    layers_path = (LAYER_BASE / tag / ("sub-%s" % sub) /
                   ("sub-%s_layers_%s_%dlayers_func.nii.gz"
                    % (sub, args.layer_method, args.nr_layers)))

    for p, name in [(atlas_path, "atlas"), (layers_path, "layers")]:
        if not p.exists():
            print("[ERROR] sub-%s: %s not found:\n  %s" % (sub, name, p))
            return sub, False

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    atlas_img,  atlas_data  = load_img(atlas_path)
    layers_img, layers_data = load_img(layers_path)

    atlas_data  = atlas_data.squeeze().astype(np.int32)
    layers_data = layers_data.squeeze().astype(np.int16)

    if atlas_data.shape != layers_data.shape:
        print("[ERROR] sub-%s: shape mismatch atlas%s vs layers%s"
              % (sub, atlas_data.shape, layers_data.shape))
        return sub, False

    gm_mask = layers_data > 0

    # ── ROI list ──────────────────────────────────────────────────────────────
    target_names, label_info = get_target_rois(args)
    if not label_info:
        print("[ERROR] sub-%s: no valid ROIs found." % sub)
        return sub, False

    # ── build non-overlap label volume ────────────────────────────────────────
    roi_label = build_roi_label(
        atlas_data, gm_mask, label_info,
        dilate=args.dilate, connectivity=args.connectivity
    )

    n_assigned = int(np.sum(roi_label > 0))
    n_gm       = int(np.sum(gm_mask))
    print("[sub-%s] %d / %d GM voxels assigned (%d ROIs)"
          % (sub, n_assigned, n_gm, len(label_info)))

    # ── save combined label volume ────────────────────────────────────────────
    save_nifti(roi_label, layers_img, out_dir / "allroi_label.nii.gz")

    # ── save per-layer label volumes ──────────────────────────────────────────
    for L in range(1, args.nr_layers + 1):
        layer_mask = layers_data == L
        layer_roi  = (roi_label * layer_mask).astype(np.int16)
        save_nifti(layer_roi, layers_img,
                   out_dir / ("layer%d_roi_label.nii.gz" % L))
        n_L = int(np.sum(layer_roi > 0))
        print("  layer %d: %d voxels assigned" % (L, n_L))

    # ── save ROI name list ────────────────────────────────────────────────────
    np.save(str(out_dir / "roi_names.npy"), np.array(target_names))

    print("[DONE] sub-%s → %s" % (sub, out_dir))
    return sub, True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS
    args_dict = vars(args)

    tag = tag_from_args(args)
    print("=" * 60)
    print(" Build ROI column masks (non-overlap)")
    print(" ROI set:    %s (%d regions)" % (
        args.roi_set,
        len([n for n in HCP_MMP_LH_LABELS if n in VALID_REGIONS])
        if args.roi_set == "valid_regions"
        else len(HCP_MMP_LH_LABELS)))
    print(" Tag:        %s" % tag)
    print(" Dilate:     %d" % args.dilate)
    print(" Subjects:   %d" % len(subjects))
    print(" Parallel:   %d" % args.parallel)
    print(" Output:     %s" % (OUT_BASE / tag))
    print("=" * 60)

    results = {}

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(process_subject, sub, args_dict): sub
                for sub in subjects
            }
            for fut in as_completed(futures):
                sub_id, ok = fut.result()
                results[sub_id] = ok
    else:
        for sub in subjects:
            _, ok = process_subject(sub, args_dict)
            results[sub] = ok

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    ok_list   = [s for s in subjects if results.get(s)]
    fail_list = [s for s in subjects if not results.get(s)]
    print(" OK:   %d — %s" % (len(ok_list),   ok_list))
    print(" FAIL: %d — %s" % (len(fail_list), fail_list))
    print("=" * 60)

    if fail_list:
        sys.exit(1)


if __name__ == "__main__":
    main()