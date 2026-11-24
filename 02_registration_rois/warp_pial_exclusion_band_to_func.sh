#!/usr/bin/env bash
# Build the T1-space outer-pial exclusion band and warp it to functional space.
#
# IMPORTANT SEMANTICS
#   output value 1 = exclude this voxel
#   output value 0 = retain this voxel
#
# The exclusion band is defined directly from the continuous depth metric:
#   (full no-rim layer map > 0) AND
#   (depth metric >= metric_max * (1 - PCT/100))
# This isolates the high-depth/pial band without also deleting metric==0 voxels.
# It deliberately does not subtract the saved no-pial layer map, because that
# map was generated with step(metric) and therefore also zeros metric==0 voxels.
#
# Usage:
#   bash warp_pial_exclusion_band_to_func.sh --subject 01
#   bash warp_pial_exclusion_band_to_func.sh
#   bash warp_pial_exclusion_band_to_func.sh --force
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"

set -euo pipefail

BIDS_ROOT="${PROJECT_ROOT}/BIDS"
DERIV_ROOT="${PROJECT_ROOT}/derivatives_0331"
REG_BASE="${DERIV_ROOT}/Step1_Anat2Func/T1w2func_MPRAGEised"
OUT_BASE="${DERIV_ROOT}/S6_validation/pial_depth_exclusion/pial_exclusion_bands_func"

SUBJECTS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 16)
ONLY_SUB=""
LD=1000
NR_LAYERS=3
METHOD="equivol"
PCT=5
LAYER_VERSION="layers_T1space_LN2_v3"
FORCE=0

usage() {
    echo "Usage: bash $(basename "$0") [--subject 01] [--pct 5] [--force]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)       ONLY_SUB="${2#sub-}"; shift 2 ;;
        --pct)           PCT="$2"; shift 2 ;;
        --ld)            LD="$2"; shift 2 ;;
        --nr-layers)     NR_LAYERS="$2"; shift 2 ;;
        --method)        METHOD="$2"; shift 2 ;;
        --layer-version) LAYER_VERSION="$2"; shift 2 ;;
        --force)         FORCE=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; usage; exit 2 ;;
    esac
done

if [[ -n "${ONLY_SUB}" ]]; then
    SUBJECTS=("${ONLY_SUB}")
fi

for cmd in antsApplyTransforms 3dcalc 3dBrickStat; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${cmd}"
        exit 1
    fi
done

TAG="ld${LD}_${NR_LAYERS}layers_${METHOD}"
OUT_ROOT="${OUT_BASE}/${TAG}_pial${PCT}pct"
mkdir -p "${OUT_ROOT}"
QC_FILE="${OUT_ROOT}/pial_exclusion_warp_qc.csv"
QC_TMP="$(mktemp /tmp/pial_warp_qc_XXXXXX.csv)"
trap 'rm -f "${QC_TMP}"' EXIT
echo "subject,pct,metric_max,depth_threshold,t1_full_no_rim_voxels,t1_exclusion_voxels,t1_exclusion_fraction,func_exclusion_voxels,t1_layer1_excluded,t1_layer2_excluded,t1_layer3_excluded,func_range_min,func_range_max,status" > "${QC_TMP}"

count_nonzero() {
    3dBrickStat -non-zero -count "$1" 2>/dev/null | awk '{print $1}'
}

OK=0
FAIL=0

for SUB in "${SUBJECTS[@]}"; do
    SUBJECT="sub-${SUB}"
    LAYER_DIR="${BIDS_ROOT}/${SUBJECT}/anat/T1/${SUBJECT}_OUTPUT.nii/SUMA/${LAYER_VERSION}/${TAG}"
    FULL_LAYER="${LAYER_DIR}/layers_${METHOD}_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    METRIC_NII="${LAYER_DIR}/LN2_${METHOD}_${NR_LAYERS}layers_metric_${METHOD}.nii"
    METRIC_GZ="${METRIC_NII}.gz"
    METRIC_STD="${LAYER_DIR}/${METHOD}_metric.nii.gz"
    FUNC_REF="${BIDS_ROOT}/${SUBJECT}/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz"
    AFFINE="${REG_BASE}/${SUBJECT}/${SUBJECT}_MPRAGEisedT1w2func_0GenericAffine.mat"
    SUB_OUT="${OUT_ROOT}/${SUBJECT}"
    T1_BAND="${SUB_OUT}/${SUBJECT}_pial_outer${PCT}pct_exclusion_T1.nii.gz"
    FUNC_BAND="${SUB_OUT}/${SUBJECT}_pial_outer${PCT}pct_exclusion_func.nii.gz"
    mkdir -p "${SUB_OUT}"

    echo ""
    echo "[${SUBJECT}] Building outer-pial ${PCT}% exclusion band"

    MISSING=0
    METRIC=""
    for candidate in "${METRIC_NII}" "${METRIC_GZ}" "${METRIC_STD}"; do
        if [[ -f "${candidate}" ]]; then
            METRIC="${candidate}"
            break
        fi
    done

    for f in "${FULL_LAYER}" "${FUNC_REF}" "${AFFINE}"; do
        if [[ ! -f "${f}" ]]; then
            echo "  [ERROR] Missing: ${f}"
            MISSING=1
        fi
    done
    if [[ -z "${METRIC}" ]]; then
        echo "  [ERROR] Continuous ${METHOD} depth metric not found in ${LAYER_DIR}"
        MISSING=1
    fi
    if [[ "${MISSING}" -eq 1 ]]; then
        echo "${SUBJECT},${PCT},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,FAIL_MISSING" >> "${QC_TMP}"
        ((FAIL++)) || true
        continue
    fi

    METRIC_MAX="$(3dBrickStat -slow -non-zero -max "${METRIC}" 2>/dev/null | awk 'NF{print $1; exit}')"
    if [[ -z "${METRIC_MAX}" ]] || ! awk "BEGIN{exit !(${METRIC_MAX}+0 > 0)}"; then
        echo "  [ERROR] Could not determine a positive metric maximum: ${METRIC}"
        echo "${SUBJECT},${PCT},${METRIC_MAX:-NA},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,FAIL_METRIC_MAX" >> "${QC_TMP}"
        ((FAIL++)) || true
        continue
    fi
    THRESHOLD="$(awk "BEGIN{printf \"%.9f\", ${METRIC_MAX} * (1.0 - ${PCT}/100.0)}")"

    if [[ "${FORCE}" -eq 1 || ! -f "${T1_BAND}" ]]; then
        3dcalc \
            -a "${METRIC}" \
            -b "${FULL_LAYER}" \
            -expr "step(b)*step(a-${THRESHOLD})" \
            -prefix "${T1_BAND}" \
            -datum byte \
            -overwrite >/dev/null
    fi

    T1_FULL_COUNT="$(count_nonzero "${FULL_LAYER}")"
    T1_COUNT="$(count_nonzero "${T1_BAND}")"
    T1_FRACTION="$(awk "BEGIN{if (${T1_FULL_COUNT}+0 > 0) printf \"%.8f\", ${T1_COUNT}/${T1_FULL_COUNT}; else print \"NA\"}")"
    if [[ -z "${T1_COUNT}" || "${T1_COUNT}" -eq 0 ]]; then
        echo "  [ERROR] T1 exclusion band is empty."
        echo "${SUBJECT},${PCT},${METRIC_MAX},${THRESHOLD},${T1_FULL_COUNT:-NA},${T1_COUNT:-NA},${T1_FRACTION:-NA},NA,NA,NA,NA,NA,NA,FAIL_EMPTY_T1" >> "${QC_TMP}"
        ((FAIL++)) || true
        continue
    fi

    TMP_DIR="$(mktemp -d /tmp/pial_layer_count_XXXXXX)"
    LAYER_COUNTS=()
    for L in $(seq 1 "${NR_LAYERS}"); do
        TMP_L="${TMP_DIR}/layer${L}.nii.gz"
        3dcalc -a "${FULL_LAYER}" -b "${T1_BAND}" \
            -expr "equals(a,${L})*step(b)" -prefix "${TMP_L}" \
            -datum byte -overwrite >/dev/null
        LAYER_COUNTS+=("$(count_nonzero "${TMP_L}")")
    done
    rm -rf "${TMP_DIR}"

    if [[ "${FORCE}" -eq 1 || ! -f "${FUNC_BAND}" ]]; then
        antsApplyTransforms \
            -d 3 \
            -i "${T1_BAND}" \
            -r "${FUNC_REF}" \
            -t "${AFFINE}" \
            -o "${FUNC_BAND}" \
            -n NearestNeighbor \
            --verbose 0
    fi

    if [[ ! -f "${FUNC_BAND}" ]]; then
        echo "  [ERROR] Functional-space output was not created."
        echo "${SUBJECT},${PCT},${METRIC_MAX},${THRESHOLD},${T1_FULL_COUNT},${T1_COUNT},${T1_FRACTION},NA,${LAYER_COUNTS[0]},${LAYER_COUNTS[1]},${LAYER_COUNTS[2]},NA,NA,FAIL_WARP" >> "${QC_TMP}"
        ((FAIL++)) || true
        continue
    fi

    FUNC_COUNT="$(count_nonzero "${FUNC_BAND}")"
    # Query min/max separately: some AFNI builds emit no text when both
    # statistic flags are supplied in the same 3dBrickStat invocation.
    # NIfTI headers do not always contain usable extrema, so force AFNI to
    # read all voxels instead of relying on the quick header path.
    RANGE_MIN="$(3dBrickStat -slow -min "${FUNC_BAND}" 2>/dev/null | awk 'NF{print $1; exit}')"
    RANGE_MAX="$(3dBrickStat -slow -max "${FUNC_BAND}" 2>/dev/null | awk 'NF{print $1; exit}')"
    STATUS="PASS"
    if [[ -z "${FUNC_COUNT}" || "${FUNC_COUNT}" -eq 0 ]]; then
        STATUS="FAIL_EMPTY_FUNC"
    elif [[ -z "${RANGE_MIN}" || -z "${RANGE_MAX}" ]]; then
        STATUS="FAIL_RANGE_MISSING"
    elif ! awk "BEGIN{exit !((${RANGE_MIN}+0 >= 0) && (${RANGE_MAX}+0 <= 1))}"; then
        STATUS="FAIL_NONBINARY_RANGE"
    elif [[ "${LAYER_COUNTS[0]}" -gt 0 || "${LAYER_COUNTS[1]}" -gt 0 ]]; then
        STATUS="WARN_EXCLUSION_NOT_ONLY_LAYER3"
    fi

    echo "  Metric max/threshold:  ${METRIC_MAX} / ${THRESHOLD}"
    echo "  T1 exclusion voxels:   ${T1_COUNT}/${T1_FULL_COUNT} (${T1_FRACTION})"
    echo "  By original layer:     L1=${LAYER_COUNTS[0]} L2=${LAYER_COUNTS[1]} L3=${LAYER_COUNTS[2]}"
    echo "  Func exclusion voxels: ${FUNC_COUNT}; range=[${RANGE_MIN}, ${RANGE_MAX}]"
    echo "  Status: ${STATUS}"
    echo "${SUBJECT},${PCT},${METRIC_MAX},${THRESHOLD},${T1_FULL_COUNT},${T1_COUNT},${T1_FRACTION},${FUNC_COUNT},${LAYER_COUNTS[0]},${LAYER_COUNTS[1]},${LAYER_COUNTS[2]},${RANGE_MIN},${RANGE_MAX},${STATUS}" >> "${QC_TMP}"

    if [[ "${STATUS}" == FAIL_* ]]; then
        ((FAIL++)) || true
    else
        ((OK++)) || true
    fi
done

mv "${QC_TMP}" "${QC_FILE}"
trap - EXIT

echo ""
echo "=============================================================================="
echo "Pial-band warp complete: OK/WARN=${OK}, FAIL=${FAIL}"
echo "QC: ${QC_FILE}"
echo "Output semantics: 1=exclude outer-pial band; 0=retain"
echo "=============================================================================="

[[ "${FAIL}" -gt 0 ]] && exit 1
exit 0
