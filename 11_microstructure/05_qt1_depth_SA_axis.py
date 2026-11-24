#!/usr/bin/env python3
"""
Depth-resolved qT1, native MPC gradients and the sensorimotor-association (S-A) axis.

Reproduces the structural analyses added in November 2026:
  A. Correspondence of group-mean qT1 with S-A ranks at each of the 14 micapipe
     surfaces and in depth bins (several literature schemes), with spin tests,
     cortical-thickness control and agreement with the HCP T1w/T2w map.
  B. Native (unrotated) MPC gradients, whole cortex and slab-only, against S-A ranks.
  C. Within the 96 slab regions: qT1 bins and MPC G1 against the group functional G1.

Inputs
  --v3-dir   output of compute_mpc_gradient_only_micapipe3.py (v3.1)
  --mpc-base micapipe derivatives (raw MPC matrices, for the slab-only embedding)
  --aux-dir  reference tables shipped with this script:
               SA_axis_ranks_glasser180.csv          Sydnor et al. (2021) final ranks
               sphere_centroids_fsLR_left180.csv     parcel centroids on the fsLR sphere
               hcp_reference_maps_glasser180.csv     HCP T1w/T2w, thickness, canonical gradients
               own_functional_G1_96.csv              group functional G1 (Bianca/Serio reference)
Spin tests: 1,000 rotations of left-hemisphere parcel centroids, Hungarian reassignment,
two-sided p = (1 + #|null| >= |obs|) / 1,001; the rotated map is the structural map
(qT1 or MPC); for slab analyses the rotated map is subset to the 96 regions afterwards.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, rankdata, skew
from scipy.optimize import linear_sum_assignment
from brainspace.gradient import GradientMaps

import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project")  # see config/paths.sh

warnings.filterwarnings("ignore")
DERIV = Path(PROJECT_ROOT + "/derivatives_0331")
SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "16"]
EVC = ["V1", "V2", "V3"]
PPC = ["LIPv", "LIPd", "VIP", "MIP", "IP0", "IP1", "IP2", "7PC", "7Am", "7AL", "7PL", "7Pm", "7m"]
CAUPFC = ["6d", "6v", "6mp", "FEF", "PEF", "55b", "SFL"]
N_SURF = 14
FRAC = (np.arange(N_SURF) + 1) / 15          # depth of surface k from the pial surface (16 surfaces, 2 dropped)


def depth_schemes():
    f, e = FRAC, 1e-9
    w = lambda lo, hi: list(np.where((f >= lo - e) & (f <= hi + e))[0])
    return {
        "B_thirds_of_thickness": {"superficial": w(0, 1 / 3), "middle": w(1 / 3 + 1e-6, 2 / 3), "deep": w(2 / 3 + 1e-6, 1)},
        "A_surface_thirds_5-4-5": {"superficial": list(range(0, 5)), "middle": list(range(5, 9)), "deep": list(range(9, 14))},
        "C_Sydnor2025_20-30_40-60_70-80": {"superficial": w(.2, .3), "middle": w(.4, .6), "deep": w(.7, .8)},
        "D_single_depths_quarter_half_threequarter": {"superficial": [int(np.argmin(abs(f - .25)))], "middle": [6, 7],
                                                      "deep": [int(np.argmin(abs(f - .75)))]},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v3-dir", type=Path, default=DERIV / "Step4_MPC_G1" / "Nov" / "mpc_gradient_only_micapipe3")
    p.add_argument("--mpc-base", type=Path, default=Path(PROJECT_ROOT + "/derivatives/micapipe_v0.2.0"))
    p.add_argument("--aux-dir", type=Path, default=Path(__file__).resolve().parent / "aux")
    p.add_argument("--out-dir", type=Path, default=DERIV / "Step4_MPC_G1" / "Nov" / "qt1_depth_SA_axis")
    p.add_argument("--n-spins", type=int, default=1000)
    p.add_argument("--spin-seed", type=int, default=42)
    return p.parse_args()


# ---------------- spin permutations ----------------
def rand_rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def make_spins(c, n, seed):
    rng = np.random.default_rng(seed)
    P = np.empty((n, len(c)), int)
    for k in range(n):
        cr = c @ rand_rot(rng).T
        P[k] = linear_sum_assignment(-(c @ cr.T))[1]
    return P


def rank_resid(x, z):
    rx, rz = rankdata(x), rankdata(z)
    return rx - np.polyval(np.polyfit(rz, rx, 1), rz)


class Spin:
    def __init__(self, perms):
        self.P = perms

    def test(self, x, y, idx=None, partial=None):
        """x is rotated (length 180), y fixed; optional subset idx and rank-partial covariate."""
        idx = np.arange(len(x)) if idx is None else np.asarray(idx)
        if partial is None:
            f = lambda a, b: spearmanr(a, b)[0]
        else:
            z = partial[idx]
            f = lambda a, b: np.corrcoef(rank_resid(a, z), rank_resid(b, z))[0, 1]
        r = f(x[idx], y[idx])
        null = np.array([f(x[p][idx], y[idx]) for p in self.P])
        return dict(rho=float(r), p_spin=float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))),
                    null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)))


# ---------------- MPC ----------------
def load_mpc(base, sub, lh, rh, regions):
    fp = base / f"sub-{sub}" / "mpc" / "acq-T1map" / f"sub-{sub}_atlas-glasser-360_desc-MPC.shape.gii"
    raw = np.squeeze(np.array([np.asarray(d.data) for d in nib.load(str(fp)).darrays])).astype(float)
    up = np.triu(raw, 1); m = up + up.T
    keep = [i for i in range(362) if i not in (0, 181)]
    return m[np.ix_(keep, keep)]


def embed(M, seed=0, ref=None, sparsity=0.9):
    gm = GradientMaps(n_components=10, kernel="normalized_angle", approach="dm", random_state=seed,
                      alignment="procrustes" if ref is not None else None)
    gm.fit(M, sparsity=sparsity, alpha=0.5, diffusion_time=0, reference=ref)
    return (gm.aligned_ if ref is not None else gm.gradients_), gm.lambdas_


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sa = pd.read_csv(a.aux_dir / "SA_axis_ranks_glasser180.csv")
    L = list(sa.region); SA = sa.SA_rank.values
    cen = pd.read_csv(a.aux_dir / "sphere_centroids_fsLR_left180.csv").set_index("region").loc[L, ["x", "y", "z"]].values
    hcp = pd.read_csv(a.aux_dir / "hcp_reference_maps_glasser180.csv").set_index("region").loc[L]
    own = pd.read_csv(a.aux_dir / "own_functional_G1_96.csv").set_index("region")
    R96 = list(own.index); i96 = [L.index(r) for r in R96]
    ownG1 = np.full(180, np.nan); ownG1[i96] = own.own_functional_G1.values
    spin = Spin(make_spins(cen, a.n_spins, a.spin_seed))

    # ---------- qT1 profiles from v3 ----------
    d = pd.read_csv(a.v3_dir / "subject_roi_intensity_moments_long.csv")
    surf_cols = [f"qT1_surface{k:02d}_from_pial" for k in range(1, N_SURF + 1)]
    missing = sorted(set(L) - set(d.roi))
    if missing:
        raise ValueError(f"Regions missing from v3 output (rerun v3.1, which adds H): {missing}")
    subs = sorted(d.subject.unique())
    prof = np.stack([d[d.subject == s].set_index("roi").loc[L, surf_cols].values.T for s in subs])   # sub x 14 x 180
    G = prof.mean(0)
    thick_own = d.groupby("roi").thickness.mean().reindex(L).values
    if not np.isfinite(thick_own).all():
        raise ValueError("Own thickness missing for some regions; run v3.1 with --thickness-mode glasser")
    if not (G[0] > G[-1]).mean() > 0.9:
        raise RuntimeError("Depth orientation check failed (qT1 should decrease from pial to white side)")

    # A1: per-surface correspondence
    rows = []
    for k in range(N_SURF):
        t = spin.test(G[k], SA)
        rows.append(dict(surface_from_pial=k + 1, depth_fraction=FRAC[k], mean_qT1=G[k].mean(), **t,
                         rho_HCP_T1wT2w=spearmanr(G[k], hcp.HCP_T1wT2w)[0]))
    pd.DataFrame(rows).to_csv(a.out_dir / "A1_per_surface_qT1_vs_SA.csv", index=False)

    # A2: depth bins under several schemes
    rows = []
    for scheme, bins in depth_schemes().items():
        for comp, idx in bins.items():
            x = G[idx].mean(0)
            subj = np.array([spearmanr(prof[i, idx].mean(0), SA)[0] for i in range(len(subs))])
            rows.append(dict(
                scheme=scheme, bin=comp, surfaces=f"{min(idx) + 1}-{max(idx) + 1}",
                depth=f"{100 * FRAC[min(idx)]:.0f}-{100 * FRAC[max(idx)]:.0f}%",
                **{f"SA_{k}": v for k, v in spin.test(x, SA).items()},
                **{f"SA_partial_own_thickness_{k}": v for k, v in spin.test(x, SA, partial=thick_own).items() if k in ("rho", "p_spin")},
                **{f"SA_partial_HCP_thickness_{k}": v for k, v in spin.test(x, SA, partial=hcp.HCP_thickness.values).items() if k in ("rho", "p_spin")},
                **{f"T1wT2w_{k}": v for k, v in spin.test(x, hcp.HCP_T1wT2w.values).items() if k in ("rho", "p_spin")},
                **{f"ownG1_slab96_{k}": v for k, v in spin.test(x, ownG1, idx=i96).items() if k in ("rho", "p_spin")},
                subject_rho_mean=subj.mean(), subject_rho_min=subj.min(), subject_rho_max=subj.max(),
                subjects_positive=int((subj > 0).sum())))
    for nm, x in (("whole_profile_mean", G.mean(0)), ("whole_profile_skewness", skew(G, axis=0))):
        rows.append(dict(scheme="E_whole_profile", bin=nm, surfaces="1-14", depth="7-93%",
                         **{f"SA_{k}": v for k, v in spin.test(x, SA).items()},
                         **{f"SA_partial_own_thickness_{k}": v for k, v in spin.test(x, SA, partial=thick_own).items() if k in ("rho", "p_spin")},
                         **{f"T1wT2w_{k}": v for k, v in spin.test(x, hcp.HCP_T1wT2w.values).items() if k in ("rho", "p_spin")}))
    pd.DataFrame(rows).to_csv(a.out_dir / "A2_depth_bins_vs_SA_by_scheme.csv", index=False)
    pd.DataFrame({"region": L, "SA_rank": SA, "own_thickness": thick_own,
                  **{f"qT1_surface{k + 1:02d}": G[k] for k in range(N_SURF)}}).to_csv(a.out_dir / "A3_group_qT1_profiles_180.csv", index=False)

    # ---------- B: native MPC gradients ----------
    lab = pd.read_csv(a.v3_dir / "atlas_label_check.csv").set_index("roi").loc[L]
    lh, rh = lab.lh_col.values.astype(int), lab.rh_col.values.astype(int)
    Z = np.stack([load_mpc(a.mpc_base, s.replace("sub-", ""), lh, rh, L) for s in subs])
    grp = Z.mean(0)
    g, lam = embed(grp)
    var = lam / lam.sum()
    bil = (g[lh] + g[rh]) / 2                                            # 180 x 10, bilateral
    rows = []
    for k in range(3):
        x = bil[:, k] * np.sign(spearmanr(bil[:, k], SA)[0] or 1)
        rows.append(dict(analysis="whole_cortex", component=f"G{k + 1}", variance_explained=var[k],
                         **{f"SA_{q}": v for q, v in spin.test(x, SA).items()},
                         **{f"SA_slab96_{q}": v for q, v in spin.test(x, SA, idx=i96).items() if q in ("rho", "p_spin")},
                         **{f"ownG1_slab96_{q}": v for q, v in spin.test(x, ownG1, idx=i96).items() if q in ("rho", "p_spin")},
                         rho_canonical_MPC_G1=spearmanr(x, hcp.canonical_MPC_G1)[0], rho_HCP_T1wT2w=spearmanr(x, hcp.HCP_T1wT2w)[0],
                         rho_left_right=float(np.corrcoef(g[lh, k], g[rh, k])[0, 1])))
    # slab-only re-embedding (bilateral-averaged 96 x 96 matrix)
    iL, iR = lh[i96], rh[i96]
    B = (grp[np.ix_(iL, iL)] + grp[np.ix_(iR, iR)] + grp[np.ix_(iL, iR)] + grp[np.ix_(iR, iL)]) / 4
    np.fill_diagonal(B, 0)
    gs, ls = embed(B)
    for k in range(3):
        v96 = gs[:, k]; full = np.full(180, np.nan); full[i96] = v96
        t = spin.test(SA, full, idx=i96); r, p = t["rho"], t["p_spin"]   # rotate S-A, subset
        rows.append(dict(analysis="slab_only_96", component=f"G{k + 1}", variance_explained=ls[k] / ls.sum(),
                         SA_rho=abs(r), SA_p_spin=p, rho_with_whole_cortex_G1=abs(spearmanr(v96, bil[i96, 0])[0])))
    pd.DataFrame(rows).to_csv(a.out_dir / "B1_native_MPC_gradients_vs_SA.csv", index=False)
    pd.DataFrame({"component": np.arange(1, 11), "lambda": lam, "variance_explained": var}).to_csv(a.out_dir / "B2_native_MPC_lambdas.csv", index=False)
    # individual gradients (Procrustes to group) and leave-one-out group gradients
    rows = []
    for i, s in enumerate(subs):
        gi, li = embed(Z[i], ref=g)
        b = (gi[lh, 0] + gi[rh, 0]) / 2
        go, _ = embed(np.delete(Z, i, 0).mean(0))
        bo = (go[lh, 0] + go[rh, 0]) / 2
        rows.append(dict(subject=s, variance_G1=li[0] / li.sum(), rho_G1_SA=spearmanr(b, SA)[0],
                         p_spin_G1_SA=spin.test(b, SA)["p_spin"], rho_with_group_G1=spearmanr(b, bil[:, 0])[0],
                         leave_out_group_G1_rho_SA=abs(spearmanr(bo, SA)[0])))
    pd.DataFrame(rows).to_csv(a.out_dir / "B3_individual_and_leave_one_out_MPC_G1.csv", index=False)
    pd.DataFrame({"region": L, "SA_rank": SA, **{f"MPC_G{k + 1}": bil[:, k] for k in range(3)}}).to_csv(
        a.out_dir / "B4_native_group_MPC_gradients_180.csv", index=False)

    # composite positions on deeper qT1 (scheme B) and MPC G1
    deep = G[depth_schemes()["B_thirds_of_thickness"]["deep"]].mean(0)
    pos = []
    for nm, x in (("deeper_qT1", deep), ("MPC_G1", bil[:, 0])):
        pr = pd.Series(x, index=L).rank(pct=True)
        for c, rr in (("EVC", EVC), ("PPC", PPC), ("cauPFC", CAUPFC)):
            pos.append(dict(map=nm, composite=c, mean_value=float(pd.Series(x, index=L)[rr].mean()), mean_percentile=float(pr[rr].mean())))
    pd.DataFrame(pos).to_csv(a.out_dir / "C1_composite_positions.csv", index=False)

    json.dump(dict(subjects=subs, n_spins=a.n_spins, spin_seed=a.spin_seed, v3_dir=str(a.v3_dir),
                   depth_schemes={k: {b: [int(i) + 1 for i in v] for b, v in s.items()} for k, s in depth_schemes().items()}),
              open(a.out_dir / "analysis_metadata.json", "w"), indent=1)
    print(f"Done: {a.out_dir}")


if __name__ == "__main__":
    main()
