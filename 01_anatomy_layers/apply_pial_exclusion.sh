#!/usr/bin/env bash
# ======================================================================
# Post-processing: pial-depth exclusion for GE-BOLD vein mitigation
# ----------------------------------------------------------------------
# Reads the continuous equivol metric from LN2_LAYERS and removes the
# outermost N% of the cortex (closest to the pial surface) from the
# layer label file.
#
# Prerequisite: layers_seg_T1_runnable_v3.sh (or v4) has already run.
#
# Usage:
#   bash apply_pial_exclusion.sh                        # all subjects, 5%
#   bash apply_pial_exclusion.sh --subject sub-01       # one subject
#   bash apply_pial_exclusion.sh --pct 10               # change threshold
#   bash apply_pial_exclusion.sh --in-name layers_T1space_LN2_v4
#
# Output (added to existing layer folder):
#   layers_equivol_3layers_no_pial5pct.nii.gz   label file, 5% removed
#   pial_depth_excl_5pct_mask.nii.gz            binary QC mask
#   equivol_metric.nii.gz                       cortical depth map
# ======================================================================
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"
set -euo pipefail

# ------------------------------ defaults ------------------------------
BIDS_ROOT="${PROJECT_ROOT}/BIDS"
LD=1000
NR_LAYERS=3
METHOD="equivol"
PCT=5
FORCE=0
ONLY_SUB=""
IN_NAME="layers_T1space_LN2_v3"   # folder produced by the main script

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)  ONLY_SUB="$2";  shift 2 ;;
        --ld)       LD="$2";        shift 2 ;;
        --pct)      PCT="$2";       shift 2 ;;
        --in-name)  IN_NAME="$2";   shift 2 ;;
        --force)    FORCE=1;        shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -n "${ONLY_SUB}" ]]; then
    SUBJECTS=("${ONLY_SUB}")
else
    SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 \
              sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-16)
fi

TAG="ld${LD}_${NR_LAYERS}layers_${METHOD}"
SUFFIX="no_pial${PCT}pct"

echo "============================================================"
echo " Pial-depth exclusion (post-processing)"
echo " Source folder:  ${IN_NAME}"
echo " Tag:            ${TAG}"
echo " Exclude top:    ${PCT}% (from pial/shallow side)"
echo " Subjects:       ${#SUBJECTS[@]}"
echo " Started:        $(date)"
echo "============================================================"

OK=0; FAIL=0

for SUB in "${SUBJECTS[@]}"; do
    SUMA_DIR="${BIDS_ROOT}/${SUB}/anat/T1/${SUB}_OUTPUT.nii/SUMA"
    OUTDIR="${SUMA_DIR}/${IN_NAME}/${TAG}"

    echo ""
    echo "--- ${SUB} ---"

    # Check required input files.
    LAYER_FILE="${OUTDIR}/layers_${METHOD}_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    METRIC_NII="${OUTDIR}/LN2_${METHOD}_${NR_LAYERS}layers_metric_${METHOD}.nii"
    METRIC_GZ="${OUTDIR}/LN2_${METHOD}_${NR_LAYERS}layers_metric_${METHOD}.nii.gz"

    # Prefer .nii (LayNii outputs uncompressed by default).
    if [[ -f "${METRIC_NII}" ]]; then
        METRIC="${METRIC_NII}"
    elif [[ -f "${METRIC_GZ}" ]]; then
        METRIC="${METRIC_GZ}"
    else
        echo "  [FAIL] Metric file not found in ${OUTDIR}"
        echo "         Looked for: $(basename "${METRIC_NII}")"
        echo "         Run the main layerification script first."
        ((FAIL++)) || true
        continue
    fi

    if [[ ! -f "${LAYER_FILE}" ]]; then
        echo "  [FAIL] Layer file not found: $(basename "${LAYER_FILE}")"
        ((FAIL++)) || true
        continue
    fi

    DST="${OUTDIR}/layers_${METHOD}_${NR_LAYERS}layers_${SUFFIX}.nii.gz"
    MASK="${OUTDIR}/pial_depth_excl_${PCT}pct_mask.nii.gz"
    METRIC_STD="${OUTDIR}/equivol_metric.nii.gz"

    if [[ -f "${DST}" && "${FORCE}" -eq 0 ]]; then
        echo "  [DONE] Output exists. Use --force to regenerate."
        echo "         ${DST}"
        ((OK++)) || true
        continue
    fi

    # ------------------------------------------------------------------
    # Get the actual metric max (LayNii v2.9.0 outputs 0–NR_LAYERS,
    # NOT 0–1). Dynamic threshold ensures we always remove exactly PCT%.
    # ------------------------------------------------------------------
    METRIC_MAX=$(3dBrickStat -max -non-zero "${METRIC}" 2>/dev/null \
                 | tr -d ' \t') || METRIC_MAX=""

    if [[ -z "${METRIC_MAX}" ]] || \
       ! awk "BEGIN{exit !(${METRIC_MAX}+0 > 0)}" 2>/dev/null; then
        echo "  [WARN] Could not read metric max; defaulting to 1."
        METRIC_MAX="1"
    fi

    THRESHOLD=$(awk "BEGIN{printf \"%.6f\", ${METRIC_MAX} * (1.0 - ${PCT}/100.0)}")

    echo "  Metric file:   $(basename "${METRIC}")"
    echo "  Metric range:  0 – ${METRIC_MAX}"
    echo "  Threshold:     ${THRESHOLD}  (keep metric < this value)"

    # Save a standardised copy of the metric.
    3dcopy "${METRIC}" "${METRIC_STD}" >/dev/null 2>&1 || true

    # Build the binary depth mask:
    #   step(a)             = 1 for non-zero metric (i.e., GM voxels)
    #   step(threshold - a) = 1 when metric < threshold (keep the deep part)
    3dcalc \
        -a "${METRIC}" \
        -expr "step(a)*step(${THRESHOLD}-a)" \
        -prefix  "${MASK}" \
        -overwrite \
        -datum short

    KEPT=$(3dBrickStat -non-zero -count "${MASK}" 2>/dev/null \
           | tr -d ' ') || KEPT="NA"
    echo "  Voxels kept in mask: ${KEPT}"

    # Apply mask to the layer label file.
    3dcalc \
        -a "${LAYER_FILE}" \
        -b "${MASK}" \
        -expr 'a*b' \
        -prefix  "${DST}" \
        -overwrite \
        -datum short

    echo "  [OK] $(basename "${DST}")"

    # Per-layer counts after exclusion.
    TMPD=$(mktemp -d /tmp/pialexcl_XXXXXX)
    echo "  Layer counts after ${PCT}% pial exclusion:"
    echo "  (layer 1=deep/WM, layer ${NR_LAYERS}=shallow/pial)"
    for ((L=1; L<=NR_LAYERS; L++)); do
        TMP_L="${TMPD}/l${L}.nii.gz"
        3dcalc -a "${DST}" -expr "equals(a,${L})" \
               -prefix "${TMP_L}" -overwrite >/dev/null 2>&1
        C=$(3dBrickStat -non-zero -count "${TMP_L}" 2>/dev/null \
            | tr -d ' ') || C="NA"
        printf "    layer %d: %s voxels\n" "${L}" "${C}"
    done
    rm -rf "${TMPD}"

    ((OK++)) || true
done

echo ""
echo "============================================================"
echo " Summary  $(date)"
echo " OK=${OK}  FAIL=${FAIL}"
echo ""
echo " Output files per subject:"
echo "   layers_${METHOD}_${NR_LAYERS}layers_${SUFFIX}.nii.gz"
echo "   pial_depth_excl_${PCT}pct_mask.nii.gz"
echo "   equivol_metric.nii.gz"
echo "============================================================"

[[ "${FAIL}" -gt 0 ]] && exit 1
exit 0