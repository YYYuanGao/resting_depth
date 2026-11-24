#!/usr/bin/env bash
# ============================================================
# Warp layer files (v3) to functional space using MPRAGEised T1w→func affine
# ------------------------------------------------------------
# Fix: WARP_LIST defined inside process_subject() to avoid
#      bash array-export failure when running in parallel.
#
# Layer source:
#   <BIDS>/sub-XX/anat/T1/sub-XX_OUTPUT.nii/SUMA/
#     layers_T1space_LN2_v3/ld${LD}_${NR_LAYERS}layers_${METHOD}/
#
# Usage:
#   bash warp_layers_to_func_v3.sh --subject 01
#   bash warp_layers_to_func_v3.sh --parallel 4
#   bash warp_layers_to_func_v3.sh --subject 01 --force
# ============================================================
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"
set -euo pipefail

# ── paths ────────────────────────────────────────────────────
BIDS_ROOT="${PROJECT_ROOT}/BIDS"
REG_BASE="${PROJECT_ROOT}/derivatives_0331/Step1_Anat2Func/T1w2func_MPRAGEised"
OUT_BASE="${PROJECT_ROOT}/derivatives_0331/Step1_Anat2Func/layers2func_MPRAGEisedAffine"

# ── defaults ─────────────────────────────────────────────────
SUBJECTS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 16)
LD=1000
NR_LAYERS=3
METHOD="equivol"
PARALLEL=1
ONLY_SUB=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)   ONLY_SUB="$2"; shift 2 ;;
        --ld)        LD="$2";        shift 2 ;;
        --nr-layers) NR_LAYERS="$2"; shift 2 ;;
        --method)    METHOD="$2";    shift 2 ;;
        --parallel)  PARALLEL="$2";  shift 2 ;;
        --force)     FORCE=1;        shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "${ONLY_SUB}" ]] && SUBJECTS=("${ONLY_SUB}")

TAG="ld${LD}_${NR_LAYERS}layers_${METHOD}"
OUT_ROOT="${OUT_BASE}/${TAG}"
mkdir -p "${OUT_ROOT}"

# ── per-subject function ──────────────────────────────────────
process_subject() {
    local SUB="$1"

    # WARP_LIST defined INSIDE the function so parallel workers each
    # get a proper local array (bash cannot export arrays across processes).
    #
    # Format: "filename|interpolation|output_suffix|required"
    #   required=yes → subject fails if this file is missing
    #   required=no  → silently skipped if missing
    local WARP_LIST=(
        "layers_${METHOD}_${NR_LAYERS}layers_incl_borders.nii.gz|NearestNeighbor|layers_${METHOD}_${NR_LAYERS}layers|yes"
        "layers_${METHOD}_${NR_LAYERS}layers_no_rim_boundary.nii.gz|NearestNeighbor|layers_${METHOD}_${NR_LAYERS}layers_no_rim_boundary|no"
        "rim4LN.nii.gz|NearestNeighbor|rim4LN|yes"
        "GM.nii.gz|NearestNeighbor|GM|no"
    )

    local SUMA_DIR="${BIDS_ROOT}/sub-${SUB}/anat/T1/sub-${SUB}_OUTPUT.nii/SUMA"
    local LAYER_DIR="${SUMA_DIR}/layers_T1space_LN2_v3/ld${LD}_${NR_LAYERS}layers_${METHOD}"
    local FUNC_REF="${BIDS_ROOT}/sub-${SUB}/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz"
    local AFFINE="${REG_BASE}/sub-${SUB}/sub-${SUB}_MPRAGEisedT1w2func_0GenericAffine.mat"
    local OUT_DIR="${OUT_ROOT}/sub-${SUB}"
    mkdir -p "${OUT_DIR}"

    echo "============================================================"
    echo "sub-${SUB}  $(date '+%H:%M:%S')"
    echo "Layer dir: ${LAYER_DIR}"
    echo "Affine:    ${AFFINE}"
    echo "Out dir:   ${OUT_DIR}"
    echo "============================================================"

    # ── pre-flight ───────────────────────────────────────────
    local MISSING=0
    for f in "${FUNC_REF}" "${AFFINE}"; do
        if [[ ! -f "${f}" ]]; then
            echo "[ERROR] Missing: ${f}"
            MISSING=1
        fi
    done
    if [[ ! -d "${LAYER_DIR}" ]]; then
        echo "[ERROR] Missing layer directory: ${LAYER_DIR}"
        MISSING=1
    fi
    if [[ ${MISSING} -eq 1 ]]; then
        echo "[FAIL] sub-${SUB}: missing prerequisites"
        return 1
    fi

    # ── warp each file ───────────────────────────────────────
    local SUB_OK=1

    for ENTRY in "${WARP_LIST[@]}"; do
        local FNAME INTERP SUFFIX REQUIRED
        FNAME="$(echo    "${ENTRY}" | cut -d'|' -f1)"
        INTERP="$(echo   "${ENTRY}" | cut -d'|' -f2)"
        SUFFIX="$(echo   "${ENTRY}" | cut -d'|' -f3)"
        REQUIRED="$(echo "${ENTRY}" | cut -d'|' -f4)"

        local IN_FILE="${LAYER_DIR}/${FNAME}"
        local OUT_FILE="${OUT_DIR}/sub-${SUB}_${SUFFIX}_func.nii.gz"

        if [[ ! -f "${IN_FILE}" ]]; then
            if [[ "${REQUIRED}" == "yes" ]]; then
                echo "[ERROR] Required input missing: ${IN_FILE}"
                SUB_OK=0
            else
                echo "[SKIP]  Optional not found: ${FNAME}"
            fi
            continue
        fi

        if [[ -f "${OUT_FILE}" && "${FORCE}" -eq 0 ]]; then
            echo "[DONE]  Exists: $(basename "${OUT_FILE}")"
            continue
        fi

        [[ "${FORCE}" -eq 1 ]] && rm -f "${OUT_FILE}"

        echo "[WARP]  $(basename "${IN_FILE}") → $(basename "${OUT_FILE}")  [${INTERP}]"

        antsApplyTransforms \
            -d 3 \
            -i "${IN_FILE}" \
            -r "${FUNC_REF}" \
            -t "${AFFINE}" \
            -o "${OUT_FILE}" \
            -n "${INTERP}" \
            --verbose 0

        if [[ ! -f "${OUT_FILE}" ]]; then
            echo "[ERROR] Output not created: ${OUT_FILE}"
            SUB_OK=0
            continue
        fi

        if command -v fslstats >/dev/null 2>&1; then
            local RANGE
            RANGE=$(fslstats "${OUT_FILE}" -R 2>/dev/null || echo "unavailable")
            echo "[OK]    $(basename "${OUT_FILE}")  range: ${RANGE}"
        else
            echo "[OK]    $(basename "${OUT_FILE}")"
        fi

        # per-layer counts for primary layer file
        if [[ "${SUFFIX}" == "layers_${METHOD}_${NR_LAYERS}layers" ]] && \
           command -v 3dBrickStat >/dev/null 2>&1; then
            echo "        Layer voxel counts in func space:"
            local L TMP COUNT
            for L in $(seq 1 "${NR_LAYERS}"); do
                TMP="/tmp/sub-${SUB}_${TAG}_L${L}_$$.nii.gz"
                3dcalc -a "${OUT_FILE}" -expr "equals(a,${L})" \
                       -prefix "${TMP}" -overwrite >/dev/null 2>&1 || true
                COUNT=$(3dBrickStat -non-zero -count "${TMP}" 2>/dev/null \
                        || echo "NA")
                rm -f "${TMP}"
                echo "          layer ${L}: ${COUNT}"
            done
        fi
    done

    if [[ ${SUB_OK} -eq 1 ]]; then
        echo "[DONE] sub-${SUB}"
        return 0
    else
        echo "[FAIL] sub-${SUB}: one or more required files failed"
        return 1
    fi
}

export -f process_subject
export BIDS_ROOT REG_BASE OUT_BASE OUT_ROOT LD NR_LAYERS METHOD TAG FORCE

# ── run ──────────────────────────────────────────────────────
echo "============================================================"
echo " Warp layers (v3) → functional space"
echo " Subjects: ${#SUBJECTS[@]}"
echo " Tag:      ${TAG}"
echo " Parallel: ${PARALLEL}"
echo " Output:   ${OUT_ROOT}"
echo " Started:  $(date)"
echo "============================================================"

FAILED=()

if [[ "${PARALLEL}" -gt 1 ]]; then
    if command -v parallel >/dev/null 2>&1; then
        printf '%s\n' "${SUBJECTS[@]}" | parallel -j "${PARALLEL}" process_subject {}
    else
        printf '%s\n' "${SUBJECTS[@]}" | \
            xargs -P "${PARALLEL}" -I{} bash -c 'process_subject "$@"' _ {}
    fi
else
    for SUB in "${SUBJECTS[@]}"; do
        process_subject "${SUB}" || FAILED+=("${SUB}")
    done
fi

# ── summary ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Summary  $(date)"
echo "============================================================"

OK=0; FAIL=0
for SUB in "${SUBJECTS[@]}"; do
    MAIN="${OUT_ROOT}/sub-${SUB}/sub-${SUB}_layers_${METHOD}_${NR_LAYERS}layers_func.nii.gz"
    RIM="${OUT_ROOT}/sub-${SUB}/sub-${SUB}_rim4LN_func.nii.gz"
    if [[ -f "${MAIN}" && -f "${RIM}" ]]; then
        echo "  [OK]   sub-${SUB}"
        ((OK++)) || true
    else
        MISSING_FILES=""
        [[ ! -f "${MAIN}" ]] && MISSING_FILES+="layers_func "
        [[ ! -f "${RIM}"  ]] && MISSING_FILES+="rim4LN_func"
        echo "  [FAIL] sub-${SUB}  (missing: ${MISSING_FILES})"
        ((FAIL++)) || true
    fi
done

echo ""
echo "Succeeded: ${OK} / ${#SUBJECTS[@]}"
echo "Failed:    ${FAIL} / ${#SUBJECTS[@]}"
echo "Output:    ${OUT_ROOT}"
echo ""
echo "QC example:"
echo "freeview \\"
echo "  ${BIDS_ROOT}/sub-01/func/derivatives/fsl_afni_fsl_preproc/AP_dc_st_mc_mean.nii.gz \\"
echo "  ${OUT_ROOT}/sub-01/sub-01_layers_${METHOD}_${NR_LAYERS}layers_func.nii.gz:colormap=heat:opacity=0.5"
echo "============================================================"

[[ ${FAIL} -gt 0 ]] && exit 1
exit 0