"""Why does one redshift solution outweigh another? Split each solution's evidence into a fit
term and a prior-volume cost per parameter, from windowed NSS runs (nss_jax.py --z-window).

For a run restricted to a redshift window w (prior renormalized inside it):
    log Z_w = <log L>_post - KL(post || prior)
and the solution's share of the full evidence is P_w Z_w, with P_w the prior mass of the
window. NSS works in the unit cube, where the prior is uniform (conditional priors such as
mass | z are inside the transform), so
    KL = sum_p KL_p - I,    KL_p = integral of q_p(u) log q_p(u) du  (q_p: posterior marginal of u_p)
and I >= 0 is the part of the cost carried by correlations between parameters. KL_p is the
prior-volume cost of parameter p: large when the data pin it down, or when the solution
sits where that parameter's prior density is low (a narrow range of u). A median u near 0
or 1 means the solution lives in the tail of that prior.

Compare two solutions of the same galaxy:
    log(weight_A / weight_B) = log(P_A/P_B) + (<log L>_A - <log L>_B) - sum_p (KL_p,A - KL_p,B) + (I_A - I_B)

  python evidence_budget.py <run_A.npz> <run_B.npz> [...]      (one galaxy; same parameters)
Each .npz is nss_jax.py output with its .json next to it (settings.z_window).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NBINS = 40


def kl_u(u, w, nbins=NBINS):
    """KL of a weighted 1-d sample on [0,1] against U(0,1), via a histogram (biased up by
    roughly (nbins-1)/(2 ESS); the same bias for every parameter and run of similar ESS)."""
    h, _ = np.histogram(np.clip(u, 0, 1 - 1e-12), bins=nbins, range=(0, 1), weights=w)
    p = h / h.sum()
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] * nbins)))


def budget(f, uz, zz):
    z = np.load(f)
    s = json.load(open(f[:-4] + ".json"))
    u, w, logl, names = z["u"], z["w"], z["logl"], [str(n) for n in z["names"]]
    w = w / w.sum()
    lz = s["logZ"]
    mean_logl = float(np.sum(w * logl))
    kl_tot = mean_logl - lz
    kls = np.array([kl_u(u[:, p], w) for p in range(u.shape[1])])
    lo, hi = s["settings"]["z_window"]
    pw = float(np.interp(hi, zz, uz) - np.interp(lo, zz, uz))
    medu = np.array([np.sum(w[np.argsort(u[:, p])].cumsum() < 0.5) for p in range(u.shape[1])])
    medu = np.array([np.sort(u[:, p])[min(k, len(w) - 1)] for p, k in enumerate(medu)])
    theta_med = [s["percentiles"][n][1] for n in names]
    return dict(file=os.path.basename(f), window=(lo, hi), logPw=np.log(pw), logZ=lz,
                logZtot=np.log(pw) + lz, mean_logl=mean_logl, max_logl=float(logl.max()),
                kl=kl_tot, kls=kls, corr=float(kls.sum() - kl_tot), medu=medu,
                theta=theta_med, names=names, ess=1 / np.sum(w ** 2))


def main():
    from nss_efsps import build
    from photometry import read_photometry
    from diagnose_nss import u_to_z_table
    files = sys.argv[1:]
    s0 = json.load(open(files[0][:-4] + ".json"))
    phot = read_photometry(s0["catalog"], [s0["id"]])[0]
    m = build(phot)
    uz, zz = u_to_z_table(m, m.col["zred"], 20001)
    B = [budget(f, uz, zz) for f in files]
    names = B[0]["names"]
    tot = np.logaddexp.reduce([b["logZtot"] for b in B])
    print(f"== {s0['id']}: {len(B)} redshift windows\n")
    print(f"   {'window':>12s} {'log P_w':>8s} {'<logL>':>8s} {'max logL':>8s} {'KL':>7s} "
          f"{'sum KL_p':>8s} {'I corr':>7s} {'log P_w Z_w':>11s} {'weight':>7s} {'ESS':>6s}")
    for b in B:
        print(f"   {b['window'][0]:5.2f}-{b['window'][1]:<5.2f} {b['logPw']:8.2f} "
              f"{b['mean_logl']:8.2f} {b['max_logl']:8.2f} {b['kl']:7.2f} {b['kls'].sum():8.2f} "
              f"{b['corr']:7.2f} {b['logZtot']:11.2f} {np.exp(b['logZtot'] - tot):7.3f} "
              f"{b['ess']:6.0f}")
    ref = B[-1]
    print(f"\n   Per-parameter prior-volume cost KL_p (nats), median u in its prior, and median "
          f"value; the last column compares each window with the last one ({ref['window'][0]}-"
          f"{ref['window'][1]}): positive = this window pays more")
    hdr = "   " + f"{'parameter':26s}" + "".join(
        f" | {b['window'][0]:.2f}-{b['window'][1]:.2f}: KL   u    value" for b in B)
    print(hdr + "   | dKL vs last")
    order = np.argsort(-np.abs(B[0]["kls"] - ref["kls"]))
    for p in order:
        row = "".join(f" | {b['kls'][p]:15.2f} {b['medu'][p]:4.2f} {b['theta'][p]:7.2f}" for b in B)
        dk = " ".join(f"{b['kls'][p] - ref['kls'][p]:+6.2f}" for b in B[:-1])
        print(f"   {names[p]:26s}{row}   | {dk}")
    print("\n   Decomposition of log(weight / weight_last):")
    for b in B[:-1]:
        d = dict(prior_z=b["logPw"] - ref["logPw"], fit=b["mean_logl"] - ref["mean_logl"],
                 kl_params=-(b["kls"].sum() - ref["kls"].sum()), corr=b["corr"] - ref["corr"])
        print(f"   {b['window']} vs {ref['window']}: total {b['logZtot'] - ref['logZtot']:+.2f} = "
              f"z prior {d['prior_z']:+.2f} + fit <logL> {d['fit']:+.2f} + per-parameter volume "
              f"{d['kl_params']:+.2f} + correlations {d['corr']:+.2f}")


if __name__ == "__main__":
    main()
