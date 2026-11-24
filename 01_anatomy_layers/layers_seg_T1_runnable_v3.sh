#!/usr/bin/env bash
# ======================================================================
# T1/SUMA-space cortical layerification with AFNI/SUMA + LayNii  [v3]
# ----------------------------------------------------------------------
# 
# Purpose
#   Generate cortical layers in structural SUMA/T1 space (e.g. 0.65 mm)
#   using FreeSurfer/SUMA surfaces -> volumetric rim -> LN2_LAYERS.
#
# Requirements
#   AFNI/SUMA: 3dcalc, 3dSurf2Vol, 3dmask_tool, 3dBrickStat, 3dcopy,
#              3dinfo, MapIcosahedron
#   LayNii:    LN2_LAYERS
#   Optional:  GNU parallel
#
# Example use
#   bash layers_seg_T1_runnable_v3.sh --subject sub-01 --ld 1000
#   bash layers_seg_T1_runnable_v3.sh --ld 1000 --parallel 4
#   bash layers_seg_T1_runnable_v3.sh --subject sub-01 --force
# ======================================================================
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"
set -euo pipefail

# ------------------------------ defaults ------------------------------
BIDS_ROOT="${PROJECT_ROOT}/BIDS"
NR_LAYERS=3
LD=1000
METHOD="equivol"       # equivol | equidist | both
PARALLEL=1
ONLY_SUB=""
FORCE=0
CLEAN_DENSE=0
DO_SMOOTH=1
USE_PIAL_EXCLUSION=0   # CHANGED: disabled by default (see header notes)
OUT_TOP_NAME="layers_T1space_LN2_v3"

usage() {
    cat <<USAGE
Usage:
  bash $(basename "$0") [options]

Options:
  --subject sub-XX        Run one subject only. Default: all
  --parallel N            Run N subjects in parallel. Default: 1
  --bids-root PATH        BIDS root. Default: ${BIDS_ROOT}
  --nr-layers N           Number of layers. Default: ${NR_LAYERS}
  --ld N                  MapIcosahedron linear density. Default: ${LD}
  --method METHOD         equivol | equidist | both. Default: ${METHOD}
  --out-name NAME         Top-level output folder name. Default: ${OUT_TOP_NAME}
  --force                 Re-run and overwrite this output folder.
  --clean-dense           Remove dense surfaces/specs before remeshing.
  --no-smooth             Skip rim012 morphological smoothing.
  --pial-exclusion        Enable pial/kissing-gyri exclusion (NOT recommended;
                          see header for why it is disabled by default).
  -h, --help              Show this help.

Layer direction (per LayNii paper, Huber et al. 2021):
  layer 1 = deepest  (nearest WM/inner boundary = label 1 in rim)
  layer N = shallowest (nearest pial/outer boundary = label 2 in rim)

Main outputs per subject:
  <SUMA>/${OUT_TOP_NAME}/ld<LD>_<N>layers_<METHOD>/
      rim012.nii.gz
      rim4LN.nii.gz           (0=bg, 1=inner/WM, 2=outer/pial, 3=GM)
      layers_equivol_<N>layers_incl_borders.nii.gz
      layers_equivol_<N>layers_no_rim_boundary.nii.gz
      processing.log
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)            ONLY_SUB="$2"; shift 2 ;;
        --parallel)           PARALLEL="$2"; shift 2 ;;
        --bids-root)          BIDS_ROOT="$2"; shift 2 ;;
        --nr-layers)          NR_LAYERS="$2"; shift 2 ;;
        --ld)                 LD="$2"; shift 2 ;;
        --method)             METHOD="$2"; shift 2 ;;
        --out-name)           OUT_TOP_NAME="$2"; shift 2 ;;
        --force)              FORCE=1; shift ;;
        --clean-dense)        CLEAN_DENSE=1; shift ;;
        --no-smooth)          DO_SMOOTH=0; shift ;;
        --pial-exclusion)     USE_PIAL_EXCLUSION=1; shift ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
    esac
done

case "${METHOD}" in
    equivol|equidist|both) ;;
    *) echo "[ERROR] --method must be equivol, equidist, or both"; exit 1 ;;
esac

# sub-15 excluded (incomplete SUMA reconstruction).
if [[ -n "${ONLY_SUB}" ]]; then
    SUBJECTS=("${ONLY_SUB}")
else
    SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 \
              sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-16)
fi

require_cmd() {
    local c="$1"
    if ! command -v "${c}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${c}"
        return 1
    fi
}

check_requirements() {
    local missing=0
    for c in 3dcalc 3dSurf2Vol 3dmask_tool 3dBrickStat 3dcopy 3dinfo \
              MapIcosahedron LN2_LAYERS; do
        require_cmd "$c" || missing=1
    done
    if [[ "${missing}" -eq 1 ]]; then
        echo "[ERROR] Missing required software. Load AFNI/SUMA and LayNii first."
        exit 1
    fi
}

find_original_spec() {
    local suma_dir="$1"
    local hemi="$2"
    local spec=""
    spec=$(find "${suma_dir}" -maxdepth 1 -type f \
        -name "*_${hemi}.spec" ! -name "std.*" ! -name "*ld*" \
        | sort | head -n 1 || true)
    [[ -n "${spec}" ]] && basename "${spec}"
}

find_dense_spec() {
    local suma_dir="$1"
    local hemi="$2"
    local ld="$3"
    local spec=""
    spec=$(find "${suma_dir}" -maxdepth 1 -type f \
        -name "std_${hemi}.ld${ld}.*${hemi}.spec" ! -name "*BOTH*" \
        | sort | head -n 1 || true)
    [[ -n "${spec}" ]] && basename "${spec}"
}

run_surf2vol_pair() {
    local spec="$1"
    local wm="$2"
    local pial="$3"
    local t1="$4"
    local out="$5"
    local p1="$6"
    local pn="$7"

    3dSurf2Vol \
        -spec "${spec}" \
        -surf_A "${wm}" \
        -surf_B "${pial}" \
        -sv "${t1}" \
        -gridset "${t1}" \
        -map_func mask \
        -f_steps 40 \
        -f_index points \
        -f_p1_fr "${p1}" \
        -f_pn_fr "${pn}" \
        -prefix "${out}" \
        -overwrite
}

standardize_layer_output() {
    local outdir="$1"
    local prefix="$2"
    local method="$3"
    local nlay="$4"

    local suffix=""
    if [[ "${method}" == "equivol" ]]; then
        suffix="equivol"
    else
        suffix="equidist"
    fi

    local layer_file=""
    local candidates=()
    shopt -s nullglob
    candidates=(
        "${prefix}_layers_${suffix}.nii.gz"
        "${prefix}_layers_${suffix}.nii"
        "${prefix}_layerbins_${suffix}.nii.gz"
        "${prefix}_layerbins_${suffix}.nii"
    )
    shopt -u nullglob

    for f in "${candidates[@]}"; do
        if [[ -f "${f}" ]]; then
            layer_file="${f}"
            break
        fi
    done

    if [[ -z "${layer_file}" ]]; then
        layer_file=$(find "${outdir}" -maxdepth 1 -type f \
            -name "$(basename "${prefix}")*${suffix}*.nii*" \
            ! -name "*metric*" ! -name "*thickness*" \
            | sort | head -n 1 || true)
    fi

    if [[ -z "${layer_file}" || ! -f "${layer_file}" ]]; then
        echo "  [ERROR] Could not find LN2_LAYERS ${method} output for: ${prefix}"
        echo "  [DEBUG] Files in ${outdir}:"
        ls -lh "${outdir}" || true
        return 1
    fi

    local std_incl="${outdir}/layers_${method}_${nlay}layers_incl_borders.nii.gz"
    local std_nob="${outdir}/layers_${method}_${nlay}layers_no_rim_boundary.nii.gz"

    3dcopy "${layer_file}" "${std_incl}" >/dev/null 2>&1

    3dcalc \
        -a "${std_incl}" \
        -expr "within(a,1,${nlay})*a" \
        -prefix "${std_nob}" \
        -overwrite \
        -datum short

    echo "  [OK] Standardized ${method} output:"
    echo "       ${std_incl}"
    echo "       ${std_nob}"
    echo "       layer 1 = deepest (WM side), layer ${nlay} = shallowest (pial side)"
}

run_ln2_method() {
    local outdir="$1"
    local method="$2"
    local nlay="$3"
    local rim="${outdir}/rim4LN.nii.gz"
    local prefix="${outdir}/LN2_${method}_${nlay}layers"

    echo "  [Step 6] LN2_LAYERS ${method} (${nlay} layers, with borders)..."
    rm -f "${prefix}"*.nii "${prefix}"*.nii.gz 2>/dev/null || true

    if [[ "${method}" == "equivol" ]]; then
        LN2_LAYERS \
            -rim "${rim}" \
            -nr_layers "${nlay}" \
            -equivol \
            -incl_borders \
            -output "${prefix}"
    else
        LN2_LAYERS \
            -rim "${rim}" \
            -nr_layers "${nlay}" \
            -equal_bins \
            -incl_borders \
            -output "${prefix}"
    fi

    standardize_layer_output "${outdir}" "${prefix}" "${method}" "${nlay}"
}

qc_one_volume() {
    local label="$1"
    local file="$2"
    if [[ -f "${file}" ]]; then
        local count minmax
        count=$(3dBrickStat -non-zero -count "${file}" 2>/dev/null || echo "NA")
        minmax=$(3dBrickStat -min -max "${file}" 2>/dev/null || echo "NA")
        printf "  %-42s count=%s  min/max=%s\n" "${label}" "${count}" "${minmax}"
    else
        printf "  %-42s MISSING\n" "${label}"
    fi
}

qc_layer_counts() {
    local file="$1"
    local nlay="$2"
    [[ ! -f "${file}" ]] && return 0
    local tmpdir=""
    tmpdir=$(mktemp -d /tmp/qc_lay_XXXXXX)
    trap 'rm -rf "${tmpdir}"' RETURN
    echo "  Layer voxel counts for $(basename "${file}"):"
    echo "  (layer 1=deep/WM side, layer ${nlay}=shallow/pial side)"
    for ((i=1; i<=nlay; i++)); do
        local tmp="${tmpdir}/l${i}.nii.gz"
        3dcalc -a "${file}" -expr "equals(a,${i})" \
               -prefix "${tmp}" -overwrite >/dev/null 2>&1
        local c
        c=$(3dBrickStat -non-zero -count "${tmp}" 2>/dev/null || echo "NA")
        printf "    layer %d: %s voxels\n" "${i}" "${c}"
    done
}

process_subject() {
    local SUB="$1"
    local SUMA_DIR="${BIDS_ROOT}/${SUB}/anat/T1/${SUB}_OUTPUT.nii/SUMA"
    local METHOD_TAG="${METHOD}"
    local OUTDIR="${SUMA_DIR}/${OUT_TOP_NAME}/ld${LD}_${NR_LAYERS}layers_${METHOD_TAG}"
    local LOG="${OUTDIR}/processing.log"

    if [[ ! -d "${SUMA_DIR}" ]]; then
        echo "[SKIP] ${SUB} -- SUMA directory not found: ${SUMA_DIR}"
        return 0
    fi

    local final_file="${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    if [[ "${METHOD}" == "equidist" ]]; then
        final_file="${OUTDIR}/layers_equidist_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    elif [[ "${METHOD}" == "both" ]]; then
        final_file="${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    fi

    if [[ -f "${final_file}" && "${FORCE}" -eq 0 ]]; then
        echo "[DONE] ${SUB} -- output exists, skipping. Use --force to rerun."
        return 0
    fi

    if [[ "${FORCE}" -eq 1 ]]; then
        rm -rf "${OUTDIR}"
    fi
    mkdir -p "${OUTDIR}"

    echo "[START] ${SUB} -- $(date '+%F %T')"

    {
    echo "=== ${SUB} started $(date) ==="
    echo "SUMA_DIR=${SUMA_DIR}"
    echo "OUTDIR=${OUTDIR}"
    echo "NR_LAYERS=${NR_LAYERS}"
    echo "METHOD=${METHOD}"
    echo "LD=${LD}"
    echo "DO_SMOOTH=${DO_SMOOTH}"
    echo "USE_PIAL_EXCLUSION=${USE_PIAL_EXCLUSION}"
    echo "FORCE=${FORCE}"
    echo "SCRIPT_VERSION=v3"
    echo "rim4LN_convention: 0=bg, 1=inner(WM), 2=outer(pial), 3=GM body"

    local T1="T1.nii.gz"
    if [[ ! -f "${SUMA_DIR}/${T1}" ]]; then
        echo "  [ERROR] Missing ${SUMA_DIR}/${T1}"
        return 1
    fi

    local SPEC_LH SPEC_RH
    SPEC_LH=$(find_original_spec "${SUMA_DIR}" lh)
    SPEC_RH=$(find_original_spec "${SUMA_DIR}" rh)
    if [[ -z "${SPEC_LH}" || -z "${SPEC_RH}" ]]; then
        echo "  [ERROR] Native lh/rh spec files not found in ${SUMA_DIR}"
        ls -1 "${SUMA_DIR}"/*.spec 2>/dev/null || true
        return 1
    fi

    for f in "lh.smoothwm.gii" "rh.smoothwm.gii" "lh.pial.gii" "rh.pial.gii"; do
        if [[ ! -f "${SUMA_DIR}/${f}" ]]; then
            echo "  [ERROR] Missing surface: ${SUMA_DIR}/${f}"
            return 1
        fi
    done

    # ------------------------------------------------------------------
    # Step 0: dense mesh generation.
    # ------------------------------------------------------------------
    echo "  [Step 0] Dense mesh generation with MapIcosahedron (ld=${LD})..."
    (
        cd "${SUMA_DIR}"
        for HEMI in lh rh; do
            local SPEC_BASE DENSE_PIAL DENSE_WM DENSE_SPEC
            if [[ "${HEMI}" == "lh" ]]; then
                SPEC_BASE="${SPEC_LH}"
            else
                SPEC_BASE="${SPEC_RH}"
            fi
            DENSE_PIAL="std_${HEMI}.ld${LD}.${HEMI}.pial.gii"
            DENSE_WM="std_${HEMI}.ld${LD}.${HEMI}.smoothwm.gii"

            if [[ "${CLEAN_DENSE}" -eq 1 ]]; then
                echo "  [Step 0] Cleaning previous dense files for ${HEMI}, ld=${LD}"
                rm -f std_${HEMI}.ld${LD}* 2>/dev/null || true
            fi

            DENSE_SPEC=$(find_dense_spec "${SUMA_DIR}" "${HEMI}" "${LD}")

            if [[ -f "${DENSE_PIAL}" && -f "${DENSE_WM}" \
                  && -n "${DENSE_SPEC}" && -f "${DENSE_SPEC}" ]]; then
                echo "  [Step 0] Dense ${HEMI} surfaces/spec already exist, skipping."
            else
                echo "  [Step 0] MapIcosahedron ${HEMI}: ${SPEC_BASE}"
                MapIcosahedron \
                    -spec "${SPEC_BASE}" \
                    -ld "${LD}" \
                    -prefix "std_${HEMI}.ld${LD}." \
                    -overwrite

                DENSE_SPEC=$(find_dense_spec "${SUMA_DIR}" "${HEMI}" "${LD}")
                if [[ ! -f "${DENSE_PIAL}" || ! -f "${DENSE_WM}" \
                      || -z "${DENSE_SPEC}" || ! -f "${DENSE_SPEC}" ]]; then
                    echo "  [ERROR] MapIcosahedron did not produce expected ${HEMI} outputs."
                    return 1
                fi
            fi
        done
    )

    # ------------------------------------------------------------------
    # Step 1: surface -> volume masks on SUMA/T1 grid.
    # ------------------------------------------------------------------
    echo "  [Step 1] Surface -> volume projection in SUMA/T1 grid..."
    (
        cd "${SUMA_DIR}"
        for HEMI in lh rh; do
            local DENSE_SPEC DENSE_PIAL DENSE_WM
            DENSE_SPEC=$(find_dense_spec "${SUMA_DIR}" "${HEMI}" "${LD}")
            DENSE_PIAL="std_${HEMI}.ld${LD}.${HEMI}.pial.gii"
            DENSE_WM="std_${HEMI}.ld${LD}.${HEMI}.smoothwm.gii"

            if [[ -z "${DENSE_SPEC}" || ! -f "${DENSE_SPEC}" ]]; then
                echo "  [ERROR] Missing dense spec for ${HEMI}"
                return 1
            fi

            echo "  [Step 1] ${HEMI}: GM ribbon (WM surface to pial)"
            run_surf2vol_pair "${DENSE_SPEC}" "${DENSE_WM}" "${DENSE_PIAL}" \
                "${T1}" "${OUTDIR}/${HEMI}.GM.nii.gz" 0.0 0.0

            echo "  [Step 1] ${HEMI}: WM support (1 unit beyond WM into white matter)"
            run_surf2vol_pair "${DENSE_SPEC}" "${DENSE_WM}" "${DENSE_PIAL}" \
                "${T1}" "${OUTDIR}/${HEMI}.WM_support.nii.gz" -1.0 0.0

            # Pial kissing exclusion: disabled by default (USE_PIAL_EXCLUSION=0).
            # Reason: LayNii (Huber 2021, Section 2.2 step 10) handles kissing
            # gyri internally; no pre-exclusion in the rim file is necessary.
            # The shell-based method was found to mark the entire outer ~10% of
            # the cortex rather than only sulcal contact points.
            if [[ "${USE_PIAL_EXCLUSION}" -eq 1 ]]; then
                echo "  [Step 1] ${HEMI}: pial kissing exclusion (USE_PIAL_EXCLUSION=1)"
                run_surf2vol_pair "${DENSE_SPEC}" "${DENSE_WM}" "${DENSE_PIAL}" \
                    "${T1}" "${OUTDIR}/${HEMI}.pial_shell_outer.nii.gz" 1.05 0.10
                run_surf2vol_pair "${DENSE_SPEC}" "${DENSE_WM}" "${DENSE_PIAL}" \
                    "${T1}" "${OUTDIR}/${HEMI}.pial_shell_inner.nii.gz" 0.90 0.0
                3dcalc \
                    -a "${OUTDIR}/${HEMI}.pial_shell_outer.nii.gz" \
                    -b "${OUTDIR}/${HEMI}.pial_shell_inner.nii.gz" \
                    -expr 'step(a)*step(b)' \
                    -prefix "${OUTDIR}/${HEMI}.pial_kissing_exclusion.nii.gz" \
                    -overwrite \
                    -datum short
            else
                # Zero mask: no voxels excluded.
                3dcalc \
                    -a "${OUTDIR}/${HEMI}.GM.nii.gz" \
                    -expr '0' \
                    -prefix "${OUTDIR}/${HEMI}.pial_kissing_exclusion.nii.gz" \
                    -overwrite \
                    -datum short
            fi
        done
    )

    # ------------------------------------------------------------------
    # Step 2: merge hemispheres.
    # ------------------------------------------------------------------
    echo "  [Step 2] Merge hemispheres..."
    for VOL in GM WM_support pial_kissing_exclusion; do
        3dcalc \
            -a "${OUTDIR}/lh.${VOL}.nii.gz" \
            -b "${OUTDIR}/rh.${VOL}.nii.gz" \
            -expr 'step(a+b)' \
            -prefix "${OUTDIR}/${VOL}.nii.gz" \
            -overwrite \
            -datum short
    done

    # ------------------------------------------------------------------
    # Step 3: build rim012: 0=background, 1=GM, 2=WM support.
    # ------------------------------------------------------------------
    echo "  [Step 3] Build rim012..."
    3dcalc \
        -a "${OUTDIR}/WM_support.nii.gz" \
        -b "${OUTDIR}/GM.nii.gz" \
        -c "${OUTDIR}/pial_kissing_exclusion.nii.gz" \
        -expr 'step(a)*(2 - step(b)*(1 + step(c)))' \
        -prefix "${OUTDIR}/rim012.nii.gz" \
        -overwrite \
        -datum short

    3dcopy "${OUTDIR}/pial_kissing_exclusion.nii.gz" \
        "${OUTDIR}/pial_kissing_gyri_exclusion_mask.nii.gz" >/dev/null 2>&1 || true

    # ------------------------------------------------------------------
    # Step 4: optional morphological smoothing.
    # ------------------------------------------------------------------
    if [[ "${DO_SMOOTH}" -eq 1 ]]; then
        echo "  [Step 4] Morphological smoothing of rim012..."

        3dcalc \
            -a "${OUTDIR}/rim012.nii.gz" \
            -expr 'step(a)' \
            -prefix "${OUTDIR}/brain_mask.nii.gz" \
            -overwrite \
            -datum short

        3dcalc \
            -a "${OUTDIR}/rim012.nii.gz" \
            -expr 'step(2-a)' \
            -prefix "${OUTDIR}/nonWM_mask.nii.gz" \
            -overwrite \
            -datum short

        3dmask_tool \
            -input "${OUTDIR}/brain_mask.nii.gz" \
            -prefix "${OUTDIR}/brain_smooth.nii.gz" \
            -dilate_input -1 1 \
            -overwrite

        3dmask_tool \
            -input "${OUTDIR}/nonWM_mask.nii.gz" \
            -prefix "${OUTDIR}/nonWM_smooth.nii.gz" \
            -dilate_input -1 1 \
            -overwrite

        3dcalc \
            -a "${OUTDIR}/brain_smooth.nii.gz" \
            -b "${OUTDIR}/nonWM_smooth.nii.gz" \
            -c "${OUTDIR}/rim012.nii.gz" \
            -expr 'step(c)*(a+step(1-b))' \
            -prefix "${OUTDIR}/rim012_smooth.nii.gz" \
            -overwrite \
            -datum short
    else
        echo "  [Step 4] Smoothing skipped; copying rim012 -> rim012_smooth"
        3dcopy "${OUTDIR}/rim012.nii.gz" \
            "${OUTDIR}/rim012_smooth.nii.gz" >/dev/null 2>&1
    fi

    # ------------------------------------------------------------------
    # Step 5: build rim4LN for LayNii.
    # ------------------------------------------------------------------
    # LayNii convention (Huber et al. 2021, Figure 2C):
    #   0 = background / irrelevant voxels
    #   1 = INNER gray matter surface (WM-facing boundary)   deep side
    #   2 = OUTER gray matter surface (pial/CSF-facing boundary)  superficial side
    #   3 = pure gray matter body
    #
    #   'equals(a,1)*3 + b*1 + c*2'   (b=pial=2, c=WM=1)  
    #
    # Result in v3:
    #   layer 1 = deepest  (nearest label 1 = WM/inner boundary)
    #   layer N = shallowest (nearest label 2 = pial/outer boundary)
    # ------------------------------------------------------------------
    echo "  [Step 5] Build rim4LN for LayNii (v3 label convention corrected)..."

    3dcalc \
        -a "${OUTDIR}/rim012_smooth.nii.gz" \
        -expr 'equals(a,1)' \
        -prefix "${OUTDIR}/rimGM.nii.gz" \
        -overwrite \
        -datum short

    3dmask_tool \
        -input "${OUTDIR}/rimGM.nii.gz" \
        -prefix "${OUTDIR}/rimGM_padded.nii.gz" \
        -dilate_input 1 \
        -overwrite

    # Pial/outer boundary: background voxels adjacent to GM
    3dcalc \
        -a "${OUTDIR}/rim012_smooth.nii.gz" \
        -b "${OUTDIR}/rimGM_padded.nii.gz" \
        -expr 'equals(a,0)*equals(b,1)' \
        -prefix "${OUTDIR}/rim_pial_boundary.nii.gz" \
        -overwrite \
        -datum short

    # WM/inner boundary: WM voxels adjacent to GM
    3dcalc \
        -a "${OUTDIR}/rim012_smooth.nii.gz" \
        -b "${OUTDIR}/rimGM_padded.nii.gz" \
        -expr 'equals(a,2)*equals(b,1)' \
        -prefix "${OUTDIR}/rim_WM_boundary.nii.gz" \
        -overwrite \
        -datum short

    # CORRECTED labels:
    #   rim_WM_boundary  (inner, b=c variable) → label 1
    #   rim_pial_boundary (outer, a=b variable) → label 2
    #   GM body                                 → label 3
    # Variable names here: a=rim012_smooth, b=rim_pial_boundary, c=rim_WM_boundary
    3dcalc \
        -a "${OUTDIR}/rim012_smooth.nii.gz" \
        -b "${OUTDIR}/rim_pial_boundary.nii.gz" \
        -c "${OUTDIR}/rim_WM_boundary.nii.gz" \
        -expr 'equals(a,1)*3 + b*1 + c*2' \
        -prefix "${OUTDIR}/rim4LN.nii.gz" \
        -overwrite \
        -datum short

    echo "  [Step 5] rim4LN labels: 1=WM(inner/deep), 2=pial(outer/shallow), 3=GM"

    # ------------------------------------------------------------------
    # Step 6: LayNii layerification.
    # ------------------------------------------------------------------
    case "${METHOD}" in
        equivol)
            run_ln2_method "${OUTDIR}" equivol "${NR_LAYERS}"
            ;;
        equidist)
            run_ln2_method "${OUTDIR}" equidist "${NR_LAYERS}"
            ;;
        both)
            run_ln2_method "${OUTDIR}" equivol "${NR_LAYERS}"
            run_ln2_method "${OUTDIR}" equidist "${NR_LAYERS}"
            ;;
    esac

    # ------------------------------------------------------------------
    # QC summary.
    # ------------------------------------------------------------------
    echo ""
    echo "  [QC] Basic volume counts"
    qc_one_volume "GM"                    "${OUTDIR}/GM.nii.gz"
    qc_one_volume "WM_support"            "${OUTDIR}/WM_support.nii.gz"
    qc_one_volume "rim012"                "${OUTDIR}/rim012.nii.gz"
    qc_one_volume "rim012_smooth"         "${OUTDIR}/rim012_smooth.nii.gz"
    qc_one_volume "rim4LN"                "${OUTDIR}/rim4LN.nii.gz"

    if [[ -f "${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz" ]]; then
        qc_one_volume "layers_equivol_no_boundary" \
            "${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
        qc_layer_counts \
            "${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz" \
            "${NR_LAYERS}"
    fi
    if [[ -f "${OUTDIR}/layers_equidist_${NR_LAYERS}layers_no_rim_boundary.nii.gz" ]]; then
        qc_one_volume "layers_equidist_no_boundary" \
            "${OUTDIR}/layers_equidist_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
        qc_layer_counts \
            "${OUTDIR}/layers_equidist_${NR_LAYERS}layers_no_rim_boundary.nii.gz" \
            "${NR_LAYERS}"
    fi

    echo "=== ${SUB} finished $(date) ==="
    } >> "${LOG}" 2>&1

    if [[ -f "${final_file}" ]]; then
        echo "[OK] ${SUB} -- output: ${OUTDIR}"
    else
        echo "[FAIL] ${SUB} -- final layer file not found. See log: ${LOG}"
        return 1
    fi
}

# ------------------------------ main ----------------------------------
check_requirements

echo "============================================================"
echo " T1/SUMA-space layerification  [v3 - label convention fixed]"
echo " Subjects:       ${#SUBJECTS[@]}"
echo " BIDS_ROOT:      ${BIDS_ROOT}"
echo " Layers:         ${NR_LAYERS}"
echo " Method:         ${METHOD}"
echo " Mesh LD:        ${LD}"
echo " Parallel:       ${PARALLEL}"
echo " Smoothing:      ${DO_SMOOTH}"
echo " Pial excl.:     ${USE_PIAL_EXCLUSION} (0=off recommended)"
echo " Force:          ${FORCE}"
echo " Output name:    ${OUT_TOP_NAME}"
echo " Layer dir.:     1=deep(WM side) ... ${NR_LAYERS}=shallow(pial side)"
echo " Started:        $(date)"
echo "============================================================"

export BIDS_ROOT NR_LAYERS LD METHOD FORCE CLEAN_DENSE DO_SMOOTH \
       USE_PIAL_EXCLUSION OUT_TOP_NAME
export -f require_cmd find_original_spec find_dense_spec \
    run_surf2vol_pair standardize_layer_output run_ln2_method \
    qc_one_volume qc_layer_counts process_subject

if [[ "${PARALLEL}" -gt 1 ]]; then
    if command -v parallel >/dev/null 2>&1; then
        printf '%s\n' "${SUBJECTS[@]}" | \
            parallel -j "${PARALLEL}" process_subject {}
    else
        printf '%s\n' "${SUBJECTS[@]}" | \
            xargs -P "${PARALLEL}" -I{} bash -c 'process_subject "$@"' _ {}
    fi
else
    for SUB in "${SUBJECTS[@]}"; do
        process_subject "${SUB}" || echo "[WARN] ${SUB} failed; continuing."
    done
fi

echo ""
echo "============================================================"
echo " Summary $(date)"
echo "============================================================"
OK=0; FAIL=0
for SUB in "${SUBJECTS[@]}"; do
    SUMA_DIR="${BIDS_ROOT}/${SUB}/anat/T1/${SUB}_OUTPUT.nii/SUMA"
    OUTDIR="${SUMA_DIR}/${OUT_TOP_NAME}/ld${LD}_${NR_LAYERS}layers_${METHOD}"
    FILE_EQV="${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    FILE_EQD="${OUTDIR}/layers_equidist_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    if [[ "${METHOD}" == "equivol" && -f "${FILE_EQV}" ]]; then
        echo "  [OK]   ${SUB}  ${FILE_EQV}"
        ((OK++)) || true
    elif [[ "${METHOD}" == "equidist" && -f "${FILE_EQD}" ]]; then
        echo "  [OK]   ${SUB}  ${FILE_EQD}"
        ((OK++)) || true
    elif [[ "${METHOD}" == "both" && -f "${FILE_EQV}" && -f "${FILE_EQD}" ]]; then
        echo "  [OK]   ${SUB}  ${OUTDIR}"
        ((OK++)) || true
    else
        echo "  [FAIL] ${SUB}  see ${OUTDIR}/processing.log"
        ((FAIL++)) || true
    fi
done
echo "OK=${OK}  FAIL=${FAIL}"
echo ""
echo " Output: ${OUT_TOP_NAME}"
echo " Layer 1 = deep (WM side), layer ${NR_LAYERS} = shallow (pial side)"
echo "============================================================"