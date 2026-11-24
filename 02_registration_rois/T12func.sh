#!/usr/bin/env bash
# ============================================================
# Register MPRAGEised T1w to functional space with ANTs
# ------------------------------------------------------------
# fixed  = mean functional EPI
# moving = MPRAGEised T1w
#
# Output:
#   ${PROJECT_ROOT}/derivatives_0331/Step1_Anat2Func/T1w2func_MPRAGEised/sub-XX/
#
# Usage:
#   bash reg_MPRAGEised_T1_to_func_ANTs.sh --subject 01
#   bash reg_MPRAGEised_T1_to_func_ANTs.sh
#   bash reg_MPRAGEised_T1_to_func_ANTs.sh --subject 01 --force
# ============================================================
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"
set -euo pipefail

BIDS_ROOT="${PROJECT_ROOT}/BIDS"
OUT_BASE="${PROJECT_ROOT}/derivatives_0331/Step1_Anat2Func/T1w2func_MPRAGEised"

SUBJECTS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 16)

ONLY_SUB=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject) ONLY_SUB="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "${ONLY_SUB}" ]] && SUBJECTS=("${ONLY_SUB}")

mkdir -p "${OUT_BASE}"

echo "============================================================"
echo "MPRAGEised T1w → functional registration"
echo "Subjects: ${#SUBJECTS[@]}"
echo "Output:   ${OUT_BASE}"
echo "Force:    ${FORCE}"
echo "Started:  $(date)"
echo "============================================================"

FAILED=()

for SUB in "${SUBJECTS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "sub-${SUB}  $(date '+%H:%M:%S')"
    echo "----------------------------------------"

    T1="${BIDS_ROOT}/sub-${SUB}/anat/T1/sub-${SUB}_T1w_MPRAGEised_Done.nii.gz"
    FUNC_MEAN="${BIDS_ROOT}/sub-${SUB}/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz"

    OUT_DIR="${OUT_BASE}/sub-${SUB}"
    mkdir -p "${OUT_DIR}"

    PREFIX="${OUT_DIR}/sub-${SUB}_MPRAGEisedT1w2func_"
    AFFINE="${PREFIX}0GenericAffine.mat"
    WARPED_T1="${OUT_DIR}/sub-${SUB}_MPRAGEised_T1w_in_func.nii.gz"
    LOG="${OUT_DIR}/sub-${SUB}_MPRAGEisedT1w2func_ants.log"

    MISSING=0
    for f in "${T1}" "${FUNC_MEAN}"; do
        if [[ ! -f "${f}" ]]; then
            echo "[ERROR] Missing input: ${f}"
            MISSING=1
        fi
    done
    if [[ ${MISSING} -eq 1 ]]; then
        echo "[FAIL] sub-${SUB}: missing input, skipping"
        FAILED+=("${SUB}")
        continue
    fi

    if [[ -f "${AFFINE}" && -f "${WARPED_T1}" && "${FORCE}" -eq 0 ]]; then
        echo "[DONE] Existing registration found."
        echo "       Affine: ${AFFINE}"
        echo "       Warped: ${WARPED_T1}"
        echo "       Use --force to re-run."
        continue
    fi

    if [[ "${FORCE}" -eq 1 ]]; then
        rm -f "${AFFINE}" "${WARPED_T1}" "${LOG}" "${PREFIX}"*
    fi

    echo "[1/1] ANTs rigid registration"
    echo "      fixed  = ${FUNC_MEAN}"
    echo "      moving = ${T1}"
    echo "      affine = ${AFFINE}"

    antsRegistration \
        --dimensionality 3 \
        --float 1 \
        --output "[${PREFIX},${WARPED_T1}]" \
        --interpolation Linear \
        --winsorize-image-intensities "[0.005,0.995]" \
        --use-histogram-matching 0 \
        --initial-moving-transform "[${FUNC_MEAN},${T1},1]" \
        --transform Rigid[0.1] \
        --metric MI["${FUNC_MEAN}","${T1}",1,32,Regular,0.25] \
        --convergence "[1000x500x250x100,1e-6,10]" \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        --verbose 1 2>&1 | tee "${LOG}"

    if [[ ! -f "${AFFINE}" || ! -f "${WARPED_T1}" ]]; then
        echo "[FAIL] sub-${SUB}: registration output not created."
        FAILED+=("${SUB}")
        continue
    fi

    echo "[OK] sub-${SUB}"
    echo "     Warped T1: ${WARPED_T1}"
    echo "     Affine:    ${AFFINE}"
    echo "     Log:       ${LOG}"
done

echo ""
echo "============================================================"
echo "Summary  $(date)"
echo "============================================================"
OK=0
FAIL=0
for SUB in "${SUBJECTS[@]}"; do
    AFFINE="${OUT_BASE}/sub-${SUB}/sub-${SUB}_MPRAGEisedT1w2func_0GenericAffine.mat"
    WARPED_T1="${OUT_BASE}/sub-${SUB}/sub-${SUB}_MPRAGEised_T1w_in_func.nii.gz"
    if [[ -f "${AFFINE}" && -f "${WARPED_T1}" ]]; then
        echo "[OK]   sub-${SUB}"
        ((OK++)) || true
    else
        echo "[FAIL] sub-${SUB}"
        ((FAIL++)) || true
    fi
done

echo ""
echo "Succeeded: ${OK} / ${#SUBJECTS[@]}"
echo "Failed:    ${FAIL} / ${#SUBJECTS[@]}"
echo "Output:    ${OUT_BASE}"
echo ""
echo "QC example:"
echo "freeview \\"
echo "  ${BIDS_ROOT}/sub-01/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz \\"
echo "  ${OUT_BASE}/sub-01/sub-01_MPRAGEised_T1w_in_func.nii.gz:opacity=0.5"
echo "============================================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
