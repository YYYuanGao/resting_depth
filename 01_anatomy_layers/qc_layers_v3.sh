#!/usr/bin/env bash
# ======================================================================
# QC script for layers_seg_T1_runnable_v3 outputs
# Usage: bash qc_layers_v3.sh [--subject sub-01] [--ld 1000]
# ======================================================================
: "${PROJECT_ROOT:?Set PROJECT_ROOT first: source config/paths.sh}"
set -uo pipefail

BIDS_ROOT="${PROJECT_ROOT}/BIDS"
SUB="sub-01"
LD=1000
NR_LAYERS=3
METHOD="equivol"
OUT_TOP_NAME="layers_T1space_LN2_v3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject) SUB="$2";  shift 2 ;;
        --ld)      LD="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SUMA_DIR="${BIDS_ROOT}/${SUB}/anat/T1/${SUB}_OUTPUT.nii/SUMA"
OUTDIR="${SUMA_DIR}/${OUT_TOP_NAME}/ld${LD}_${NR_LAYERS}layers_${METHOD}"
T1="${SUMA_DIR}/T1.nii.gz"
LOG="${OUTDIR}/processing.log"

# Temp dir for counting (fixed: initialise to "" to avoid unbound variable)
TMPDIR_QC=""
TMPDIR_QC=$(mktemp -d /tmp/qc_v3_XXXXXX)
trap '[[ -n "${TMPDIR_QC}" ]] && rm -rf "${TMPDIR_QC}"' EXIT

PASS=0; WARN=0; FAIL=0
_pass() { echo "[PASS] $1"; ((PASS++)) || true; }
_warn() { echo "[WARN] $1"; ((WARN++)) || true; }
_fail() { echo "[FAIL] $1"; ((FAIL++)) || true; }

count_nonzero() {
    local f="$1"
    local c
    c=$(3dBrickStat -non-zero -count "${f}" 2>/dev/null) || c=0
    c=$(echo "${c}" | tr -d ' \t')
    [[ "${c}" =~ ^[0-9]+$ ]] || c=0
    echo "${c}"
}

count_label() {
    local file="$1" label="$2"
    local tmp="${TMPDIR_QC}/lbl${label}_$$.nii.gz"
    3dcalc -a "${file}" -expr "equals(a,${label})" \
           -prefix "${tmp}" -overwrite 2>/dev/null || { echo 0; return; }
    count_nonzero "${tmp}"
}

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

echo "======================================================"
echo " QC: ${SUB}  LD=${LD}  NR_LAYERS=${NR_LAYERS}  [v3]"
echo " OUTDIR: ${OUTDIR}"
echo " $(date '+%F %T')"
echo "======================================================"

# ---------------------------------------------------------------
# 1. Directory and log
# ---------------------------------------------------------------
echo ""
echo "--- 1. Directory & log ---"

if [[ ! -d "${OUTDIR}" ]]; then
    _fail "Output directory missing: ${OUTDIR}"
    echo "Cannot continue."; exit 1
fi
_pass "Output directory exists"

if [[ ! -f "${LOG}" ]]; then
    _fail "processing.log not found"
else
    _pass "processing.log found"
    if grep -qiE "^\s*\[ERROR\]|Step [0-9].*ERROR|Missing surface|Missing.*nii" \
            "${LOG}" 2>/dev/null; then
        _fail "Errors found in log:"
        grep -iE "^\s*\[ERROR\]|Step [0-9].*ERROR|Missing" \
            "${LOG}" 2>/dev/null | head -5 | sed 's/^/       /'
    else
        _pass "No fatal errors in processing.log"
    fi
    START=$(grep "started" "${LOG}" | head -1 | sed 's/.*started //' || echo "?")
    END=$(grep  "finished" "${LOG}" | tail -1 | sed 's/.*finished //' || echo "not finished")
    echo "       Start : ${START}"
    echo "       End   : ${END}"
fi

# ---------------------------------------------------------------
# 2. Required files
# ---------------------------------------------------------------
echo ""
echo "--- 2. Required file existence ---"

REQUIRED=(
    "rim012.nii.gz"
    "rim012_smooth.nii.gz"
    "rim4LN.nii.gz"
    "GM.nii.gz"
    "WM_support.nii.gz"
    "layers_equivol_${NR_LAYERS}layers_incl_borders.nii.gz"
    "layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
    "LN2_equivol_${NR_LAYERS}layers_metric_equivol.nii"
)
ALL_EXIST=1
for f in "${REQUIRED[@]}"; do
    if [[ -f "${OUTDIR}/${f}" ]]; then
        SIZE=$(du -sh "${OUTDIR}/${f}" 2>/dev/null | cut -f1)
        _pass "${f}  (${SIZE})"
    else
        _fail "MISSING: ${f}"
        ALL_EXIST=0
    fi
done
[[ "${ALL_EXIST}" -eq 0 ]] && { echo "[ABORT] Missing files."; exit 1; }

# ---------------------------------------------------------------
# 3. rim012 label sanity {0,1,2}
# ---------------------------------------------------------------
echo ""
echo "--- 3. rim012 sanity ---"

RIM012="${OUTDIR}/rim012.nii.gz"
R_MIN=$(3dBrickStat -min "${RIM012}" 2>/dev/null | tr -d ' ') || R_MIN="?"
R_MAX=$(3dBrickStat -max "${RIM012}" 2>/dev/null | tr -d ' ') || R_MAX="?"
echo "       rim012 min/max = ${R_MIN} / ${R_MAX}"

if [[ "${R_MIN}" == "0" && "${R_MAX}" == "2" ]]; then
    _pass "rim012 labels correct {0,1,2}"
else
    _fail "rim012 unexpected range (expected 0/2, got ${R_MIN}/${R_MAX})"
fi

GM_COUNT=0; WM_COUNT=0
for LABEL in 1 2; do
    C=$(count_label "${RIM012}" "${LABEL}")
    case ${LABEL} in
        1) echo "       label 1 (GM)         : ${C} voxels"; GM_COUNT=${C} ;;
        2) echo "       label 2 (WM support) : ${C} voxels"; WM_COUNT=${C} ;;
    esac
done

if is_int "${GM_COUNT}"; then
    if   (( GM_COUNT < 500000 )); then
        _fail  "GM count too low (${GM_COUNT})"
    elif (( GM_COUNT > 4000000 )); then
        _warn  "GM count very high (${GM_COUNT})"
    else
        _pass  "GM voxel count plausible (${GM_COUNT})"
    fi
fi

# ---------------------------------------------------------------
# 4. rim4LN label convention — v3-specific check
#    LayNii requires: 1=inner(WM), 2=outer(pial), 3=GM body
#    v2 had 1=pial, 2=WM (WRONG). v3 fixed this.
# ---------------------------------------------------------------
echo ""
echo "--- 4. rim4LN label convention (v3 fix) ---"

RIM4="${OUTDIR}/rim4LN.nii.gz"
R4_MIN=$(3dBrickStat -min "${RIM4}" 2>/dev/null | tr -d ' ') || R4_MIN="?"
R4_MAX=$(3dBrickStat -max "${RIM4}" 2>/dev/null | tr -d ' ') || R4_MAX="?"
echo "       rim4LN min/max = ${R4_MIN} / ${R4_MAX}"

if [[ "${R4_MIN}" == "0" && "${R4_MAX}" == "3" ]]; then
    _pass "rim4LN labels in range {0,1,2,3}"
else
    _fail "rim4LN unexpected range (expected 0/3, got ${R4_MIN}/${R4_MAX})"
fi

WM_BOUNDARY=0; PIAL_BOUNDARY=0; GM_BODY=0
for LABEL in 1 2 3; do
    C=$(count_label "${RIM4}" "${LABEL}")
    case ${LABEL} in
        1) echo "       label 1 (inner/WM boundary)  : ${C} voxels"; WM_BOUNDARY=${C} ;;
        2) echo "       label 2 (outer/pial boundary) : ${C} voxels"; PIAL_BOUNDARY=${C} ;;
        3) echo "       label 3 (GM body)             : ${C} voxels"; GM_BODY=${C} ;;
    esac
done

# v3 convention check: pial boundary (outer) should be LARGER than
# WM boundary (inner) because the pial surface has greater area.
if is_int "${PIAL_BOUNDARY}" && is_int "${WM_BOUNDARY}" && \
   (( WM_BOUNDARY > 0 )); then
    if (( PIAL_BOUNDARY > WM_BOUNDARY )); then
        _pass "pial boundary (label 2) > WM boundary (label 1) ✓ correct v3 convention"
    else
        _fail "pial boundary (${PIAL_BOUNDARY}) <= WM boundary (${WM_BOUNDARY})"
        echo "       This suggests labels 1 and 2 are still SWAPPED (v2 bug)."
        echo "       Expected: label 1=WM(small area), label 2=pial(large area)"
    fi
fi

if is_int "${PIAL_BOUNDARY}" && is_int "${WM_BOUNDARY}" && (( WM_BOUNDARY > 0 )); then
    RATIO=$(awk "BEGIN{printf \"%.2f\", ${PIAL_BOUNDARY}/${WM_BOUNDARY}}")
    echo "       pial/WM boundary ratio: ${RATIO}"
    if awk "BEGIN{exit !(${RATIO} > 1.0 && ${RATIO} < 2.5)}"; then
        _pass "Pial/WM ratio plausible for v3 (${RATIO})"
    else
        _warn "Pial/WM ratio unusual (${RATIO})"
    fi
fi

# ---------------------------------------------------------------
# 5. LN2_LAYERS metric sanity
# ---------------------------------------------------------------
echo ""
echo "--- 5. Equivol metric sanity ---"

METRIC="${OUTDIR}/LN2_equivol_${NR_LAYERS}layers_metric_equivol.nii"
M_MIN=$(3dBrickStat -min -non-zero "${METRIC}" 2>/dev/null | tr -d ' ') || M_MIN="?"
M_MAX=$(3dBrickStat -max           "${METRIC}" 2>/dev/null | tr -d ' ') || M_MAX="?"
echo "       Metric min(non-zero)/max = ${M_MIN} / ${M_MAX}"
echo "       Expected range: 0 – NR_LAYERS (LayNii v2.9.0 uses 0–3 scale)"

if [[ "${M_MAX}" != "?" ]]; then
    # LayNii v2.9.0 outputs metric in 0-NR_LAYERS scale (not 0-1).
    # Max should be close to NR_LAYERS (=3).
    if awk "BEGIN{exit !(${M_MAX}+0 > 2.5 && ${M_MAX}+0 <= 3.0)}"; then
        _pass "Metric max ≈ NR_LAYERS (${M_MAX}) — consistent with LayNii v2.9.0"
        echo "       Note: apply_pial_exclusion.sh uses dynamic threshold,"
        echo "             so 5% pial exclusion will be computed correctly."
    elif awk "BEGIN{exit !(${M_MAX}+0 > 0.9 && ${M_MAX}+0 <= 1.0)}"; then
        _pass "Metric max ≈ 1 (${M_MAX}) — 0-1 scale"
    else
        _warn "Metric max unusual: ${M_MAX}"
    fi
fi

# ---------------------------------------------------------------
# 6. Layer output checks
# ---------------------------------------------------------------
echo ""
echo "--- 6. Layer label file checks ---"

LAYER_FILE="${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"
LF_MIN=$(3dBrickStat -min "${LAYER_FILE}" 2>/dev/null | tr -d ' ') || LF_MIN="?"
LF_MAX=$(3dBrickStat -max "${LAYER_FILE}" 2>/dev/null | tr -d ' ') || LF_MAX="?"
echo "       Layer file min/max = ${LF_MIN} / ${LF_MAX}"

if [[ "${LF_MIN}" == "0" && "${LF_MAX}" == "${NR_LAYERS}" ]]; then
    _pass "Layer labels correct {0..${NR_LAYERS}}"
else
    _fail "Layer labels unexpected (expected 0/${NR_LAYERS}, got ${LF_MIN}/${LF_MAX})"
fi

TOTAL_LAYER=0
ALL_NONZERO=1
declare -a LAYER_COUNTS=()
for ((L=1; L<=NR_LAYERS; L++)); do
    C=$(count_label "${LAYER_FILE}" "${L}")
    LAYER_COUNTS+=("${C}")
    echo "       layer ${L}: ${C} voxels"
    if is_int "${C}"; then
        (( C == 0 )) && ALL_NONZERO=0
        TOTAL_LAYER=$(( TOTAL_LAYER + C ))
    fi
done
echo "       total labeled: ${TOTAL_LAYER}"

if [[ "${ALL_NONZERO}" -eq 1 ]]; then
    _pass "All ${NR_LAYERS} layers have voxels"
else
    _fail "One or more layers are empty!"
fi

# Layer balance check (equivol: allow ±30% of mean)
if (( TOTAL_LAYER > 0 )); then
    EXPECTED=$(( TOTAL_LAYER / NR_LAYERS ))
    BALANCED=1
    for C in "${LAYER_COUNTS[@]}"; do
        if is_int "${C}"; then
            LOW=$(( EXPECTED * 70 / 100 ))
            HIGH=$(( EXPECTED * 130 / 100 ))
            (( C < LOW || C > HIGH )) && BALANCED=0
        fi
    done
    if [[ "${BALANCED}" -eq 1 ]]; then
        _pass "Layer counts balanced (±30% of mean ~${EXPECTED})"
    else
        _warn "Layer counts outside ±30% of mean — inspect visually"
    fi
fi

# ---------------------------------------------------------------
# 7. Layer direction check (v3 fix verification)
#    With correct labels (1=WM, 2=pial), layer 1 should be deepest
#    (near WM, smaller area → fewer voxels at gyri but more in sulci).
#    For equivol: layer 3 (superficial) tends to have more voxels
#    because pial surface area > WM surface area in a folded cortex.
# ---------------------------------------------------------------
echo ""
echo "--- 7. Layer direction sanity (v3 label fix) ---"

if [[ "${#LAYER_COUNTS[@]}" -eq 3 ]] && \
   is_int "${LAYER_COUNTS[0]}" && is_int "${LAYER_COUNTS[2]}"; then
    L1="${LAYER_COUNTS[0]}"
    L3="${LAYER_COUNTS[2]}"
    echo "       layer 1 (deep/WM)      : ${L1} voxels"
    echo "       layer 3 (shallow/pial) : ${L3} voxels"
    echo "       For equivol with correct labels, layer 3 ≥ layer 1 is expected"
    echo "       (pial surface area > WM surface area in folded cortex)"
    if (( L3 >= L1 )); then
        _pass "layer 3 ≥ layer 1 ✓ consistent with correct v3 label convention"
    else
        _warn "layer 1 > layer 3 — may still be correct for this dataset;"
        echo "       confirm visually that layer 1 is near WM and layer 3 near pial"
    fi
fi

# ---------------------------------------------------------------
# 8. Geometry consistency
# ---------------------------------------------------------------
echo ""
echo "--- 8. Geometry consistency with T1 ---"

if [[ -f "${T1}" ]]; then
    T1_DIMS=$(3dinfo -n4 "${T1}" 2>/dev/null | tr -s ' ' | xargs)
    echo "       T1 dims: ${T1_DIMS}"
    for VOL in "rim012.nii.gz" "rim4LN.nii.gz" \
               "layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz"; do
        if [[ -f "${OUTDIR}/${VOL}" ]]; then
            V_DIMS=$(3dinfo -n4 "${OUTDIR}/${VOL}" 2>/dev/null | tr -s ' ' | xargs)
            if [[ "${T1_DIMS}" == "${V_DIMS}" ]]; then
                _pass "${VOL}: grid matches T1"
            else
                _fail "${VOL}: grid MISMATCH — T1=${T1_DIMS} file=${V_DIMS}"
            fi
        fi
    done
fi

# ---------------------------------------------------------------
# 9. Dense mesh reuse check
# ---------------------------------------------------------------
echo ""
echo "--- 9. Dense mesh files (reusable for batch run) ---"

for HEMI in lh rh; do
    DPIAL="${SUMA_DIR}/std_${HEMI}.ld${LD}.${HEMI}.pial.gii"
    DWM="${SUMA_DIR}/std_${HEMI}.ld${LD}.${HEMI}.smoothwm.gii"
    if [[ -f "${DPIAL}" && -f "${DWM}" ]]; then
        SP=$(du -sh "${DPIAL}" | cut -f1)
        SW=$(du -sh "${DWM}"   | cut -f1)
        _pass "${HEMI}: pial (${SP}) + smoothwm (${SW}) — will be reused for all subjects"
    else
        _warn "${HEMI}: dense mesh missing — MapIcosahedron will re-run per subject"
    fi
done

# ---------------------------------------------------------------
# Summary and verdict
# ---------------------------------------------------------------
echo ""
echo "======================================================"
echo " SUMMARY"
echo "======================================================"
printf " PASS: %-3d  WARN: %-3d  FAIL: %-3d\n" "${PASS}" "${WARN}" "${FAIL}"
echo ""

if (( FAIL == 0 && WARN == 0 )); then
    echo " VERDICT: ✓ GO — all checks passed."
    echo ""
    echo " Next steps:"
    echo "   # Fix the cosmetic tmpdir error in v3 script:"
    echo "   sed -i 's/^    local tmpdir$/    local tmpdir=\"\"/' layers_seg_T1_runnable_v3.sh"
    echo ""
    echo "   # Run all subjects:"
    echo "   bash layers_seg_T1_runnable_v3.sh --ld ${LD} --parallel 4"
    echo ""
    echo "   # Then generate 5% pial-excluded versions:"
    echo "   bash apply_pial_exclusion.sh --in-name ${OUT_TOP_NAME}"
elif (( FAIL == 0 )); then
    echo " VERDICT: ⚠ REVIEW — ${WARN} warning(s). Do visual QC below first."
else
    echo " VERDICT: ✗ STOP — ${FAIL} failure(s). Fix before batch run."
fi

# ---------------------------------------------------------------
# Visual QC commands
# ---------------------------------------------------------------
echo ""
echo "======================================================"
echo " Visual QC commands"
echo "======================================================"
cat <<QC

# 1. rim4LN on T1 — check label convention:
#    blue=1(WM/inner/deep)  red=2(pial/outer/shallow)  green=3(GM body)
freeview \\
  ${T1} \\
  ${OUTDIR}/rim4LN.nii.gz:colormap=lut:opacity=0.6

# 2. Layers on T1 — layer 1=deep(WM), layer 3=shallow(pial)
freeview \\
  ${T1} \\
  ${OUTDIR}/layers_equivol_${NR_LAYERS}layers_no_rim_boundary.nii.gz:colormap=heat:opacity=0.7

# 3. Equivol metric (0=WM/deep, max≈3=pial/shallow)
freeview \\
  ${T1} \\
  ${OUTDIR}/LN2_equivol_${NR_LAYERS}layers_metric_equivol.nii:colormap=heat:opacity=0.7

# 4. Quick value check
for f in rim012 rim4LN \\
         layers_equivol_${NR_LAYERS}layers_no_rim_boundary \\
         layers_equivol_${NR_LAYERS}layers_incl_borders; do
    printf "%-55s " "\${f}.nii.gz"
    3dBrickStat -min -max ${OUTDIR}/\${f}.nii.gz 2>/dev/null
done
QC
echo "======================================================"