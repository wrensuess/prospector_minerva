"""Build a surviving-stellar-mass table in efsps-photfit's format (efsps_photfit/survival.py,
data/surviving_fraction_mist_kroupa.npz), for any IMF. The NSS postprocessing uses a Chabrier
one, because the stellar library the w256 emulator was trained on (E-FSPS c3k_hr-mist-afe,
cc1b132) is Chabrier (imf_type 1, confirmed by Joel 2026-11-13), while efsps-photfit's table
is Kroupa.

Same construction as the efsps-photfit table (its meta: python-fsps, MIST, zcontinuous 1,
sfh 0): for each of 25 metallicities (logzsol -2.5 ... 0.5), FSPS's single-age-population
stellar_mass (stars + remnants per Msun formed) at FSPS's 107 ages (log t 5.0 ... 10.3), then
C(t) = integral_0^t f dt on 4001 log-spaced t (log t 0 ... 10.3), with f linear in log t
between ages, held at its first value below log t 5, and the trapezoid rule in linear t. With
the efsps-photfit table's own f this reproduces its C exactly (checked 2026-11-13).

Must run where python-fsps has MIST (the prosp env):
    python build_survival_table.py --imf-type 1 --out surviving_fraction_mist_chabrier.npz
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

LOGZSOL = np.linspace(-2.5, 0.5, 25)
LOG_T = np.linspace(0.0, 10.3, 4001)


def cumulative(log_t, ssp_log_ages, f):
    """C(t) = int_0^t f dt on the log_t grid (efsps-photfit's construction)."""
    t = 10.0 ** log_t
    ff = np.interp(log_t, ssp_log_ages, f)
    return ff[0] * t[0] + np.concatenate([[0.0], np.cumsum(0.5 * (ff[1:] + ff[:-1]) * np.diff(t))])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--imf-type", type=int, required=True, help="FSPS imf_type: 1 Chabrier, 2 Kroupa")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import fsps
    import os
    sp = fsps.StellarPopulation(zcontinuous=1, imf_type=args.imf_type, sfh=0)
    ages = np.asarray(sp.log_age, float)
    frac = []
    for z in LOGZSOL:
        sp.params["logzsol"] = float(z)
        frac.append(np.asarray(sp.stellar_mass, float).copy())
    frac = np.array(frac)
    cum = np.array([cumulative(LOG_T, ages, f) for f in frac])
    meta = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python_fsps": fsps.__version__, "libraries": [str(x.decode() if isinstance(x, bytes) else x).strip()
                                                         for x in sp.libraries],
            "imf_type": args.imf_type, "zcontinuous": 1, "sfh": 0,
            "SPS_HOME": os.environ.get("SPS_HOME", ""),
            "quantity": "stellar_mass (stars+remnants) per 1 Msun formed",
            "builder": "prospector_minerva nss/build_survival_table.py"}
    np.savez(args.out, logzsol=LOGZSOL, log_t=LOG_T, cumint=cum, ssp_log_ages=ages,
             ssp_fraction=frac, meta=json.dumps(meta))
    print(f"wrote {args.out}: imf_type {args.imf_type}, {frac.shape[0]} metallicities x {ages.size} ages; "
          f"f at 10 Gyr, solar: {np.interp(10.0, ages, frac[20]):.4f}")


if __name__ == "__main__":
    main()
