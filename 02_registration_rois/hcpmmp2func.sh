#!/usr/bin/env bash
# ============================================================
# Warp HCP-MMP1.0 atlas from T1 space to functional space
# using MPRAGEised T1w → functional affine
# ============================================================
#
# Purpose
# -------
# This script regenerates the HCP-MMP atlas in functional space using
# the same MPRAGEised T1w→func registration source as the layer files.
#
# Input atlas:
#   ${PROJECT_ROOT}/Atlas_HCPMMP/sub-XX_HCPMMP1.nii.gz
#
# Affine:
#   ${PROJECT_ROOT}/derivatives_0331/Step1_Anat2Func/
#     T1w2func_MPRAGEised/sub-XX/
#     sub-XX_MPRAGEisedT1w2func_0GenericAffine.mat
#
# Functional reference:
#   ${PROJECT_ROOT}/BIDS/sub-XX/func/derivatives/
#     fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz
#
# Output:
#   ${PROJECT_ROOT}/Atlas_HCPMMP/func_space_MPRAGEisedAffine/
#     sub-XX_HCPMMP1_in_func.nii.gz
#
# Interpolation:
#   NearestNeighbor, because HCP-MMP is an integer label atlas.
#
# Usage
# -----
# One subject:
#   bash warp_hcpmmp_to_func_MPRAGEisedAffine_FINAL.sh --subject 16 --force
#
# All subjects:
#   bash warp_hcpmmp_to_func_MPRAGEisedAffine_FINAL.sh --parallel 4 --force
#
# ============================================================
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"
set -euo pipefail

export AFNI_NIFTI_TYPE_WARN=NO

BIDS_ROOT="${PROJECT_ROOT}/BIDS"
ATLAS_DIR="${PROJECT_ROOT}/Atlas_HCPMMP"
OUT_DIR="${ATLAS_DIR}/func_space_MPRAGEisedAffine"
REG_BASE="${PROJECT_ROOT}/derivatives_0331/Step1_Anat2Func/T1w2func_MPRAGEised"

# sub-15 excluded: SUMA reconstruction incomplete
SUBJECTS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 16)

ONLY_SUB=""
PARALLEL=1
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)
            ONLY_SUB="$2"; shift 2 ;;
        --parallel)
            PARALLEL="$2"; shift 2 ;;
        --force)
            FORCE=1; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash warp_hcpmmp_to_func_MPRAGEisedAffine_FINAL.sh [--subject 16] [--parallel 4] [--force]"
            exit 1 ;;
    esac
done

[[ -n "${ONLY_SUB}" ]] && SUBJECTS=("${ONLY_SUB}")
mkdir -p "${OUT_DIR}"

process_subject() {
    local SUB="$1"

    local ATLAS_T1="${ATLAS_DIR}/sub-${SUB}_HCPMMP1.nii.gz"
    local AFFINE="${REG_BASE}/sub-${SUB}/sub-${SUB}_MPRAGEisedT1w2func_0GenericAffine.mat"
    local FUNC_REF="${BIDS_ROOT}/sub-${SUB}/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz"
    local OUT_FILE="${OUT_DIR}/sub-${SUB}_HCPMMP1_in_func.nii.gz"

    echo "============================================================"
    echo "sub-${SUB}  $(date '+%H:%M:%S')"
    echo "Atlas:  ${ATLAS_T1}"
    echo "Affine: ${AFFINE}"
    echo "Ref:    ${FUNC_REF}"
    echo "Out:    ${OUT_FILE}"
    echo "============================================================"

    local MISSING=0
    for f in "${ATLAS_T1}" "${AFFINE}" "${FUNC_REF}"; do
        if [[ ! -f "${f}" ]]; then
            echo "[ERROR] Missing: ${f}"
            MISSING=1
        fi
    done
    if [[ ${MISSING} -eq 1 ]]; then
        echo "[FAIL] sub-${SUB}: missing input(s)"
        return 1
    fi

    if [[ -f "${OUT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[DONE] Exists: ${OUT_FILE}"
        echo "       Use --force to overwrite."
        return 0
    fi

    [[ "${FORCE}" -eq 1 ]] && rm -f "${OUT_FILE}"

    antsApplyTransforms \
        -d 3 \
        -i "${ATLAS_T1}" \
        -r "${FUNC_REF}" \
        -t "${AFFINE}" \
        -o "${OUT_FILE}" \
        -n NearestNeighbor \
        --verbose 0

    if [[ ! -f "${OUT_FILE}" ]]; then
        echo "[FAIL] sub-${SUB}: output not created"
        return 1
    fi

    if command -v fslstats >/dev/null 2>&1; then
        local RANGE N_NONZERO
        RANGE=$(fslstats "${OUT_FILE}" -R 2>/dev/null || echo "NA NA")
        N_NONZERO=$(fslstats "${OUT_FILE}" -V 2>/dev/null | awk '{print $1}' || echo "NA")
        echo "[OK] $(basename "${OUT_FILE}")"
        echo "     Range: ${RANGE}"
        echo "     Non-zero voxels: ${N_NONZERO}"
    else
        echo "[OK] $(basename "${OUT_FILE}")"
    fi

    return 0
}

export -f process_subject
export BIDS_ROOT ATLAS_DIR OUT_DIR REG_BASE FORCE AFNI_NIFTI_TYPE_WARN

echo "============================================================"
echo "Warp HCP-MMP to func using MPRAGEised affine"
echo "Subjects: ${#SUBJECTS[@]}"
echo "Parallel: ${PARALLEL}"
echo "Output:   ${OUT_DIR}"
echo "Started:  $(date)"
echo "============================================================"

FAILED=()

if [[ "${PARALLEL}" -gt 1 ]]; then
    if command -v parallel >/dev/null 2>&1; then
        printf '%s\n' "${SUBJECTS[@]}" | parallel -j "${PARALLEL}" process_subject {}
    else
        printf '%s\n' "${SUBJECTS[@]}" | xargs -P "${PARALLEL}" -I{} bash -c 'process_subject "$@"' _ {}
    fi
else
    for SUB in "${SUBJECTS[@]}"; do
        process_subject "${SUB}" || FAILED+=("${SUB}")
    done
fi

echo ""
echo "============================================================"
echo "Summary  $(date)"
echo "============================================================"

OK=0
FAIL=0
for SUB in "${SUBJECTS[@]}"; do
    OUT_FILE="${OUT_DIR}/sub-${SUB}_HCPMMP1_in_func.nii.gz"
    if [[ -f "${OUT_FILE}" ]]; then
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
echo "Output:    ${OUT_DIR}"
echo ""
echo "QC example:"
echo "freeview \\"
echo "  ${BIDS_ROOT}/sub-16/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz \\"
echo "  ${OUT_DIR}/sub-16_HCPMMP1_in_func.nii.gz:colormap=lut:opacity=0.35"
echo "============================================================"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
