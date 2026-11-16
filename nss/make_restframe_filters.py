"""Build restframe_filters_<grid>.npz: sedpy's own AB-magnitude integration for the rest-frame
filters postprocess_parrot.py used, tabulated on an E-FSPS rest wavelength grid, so the NSS
spectra step (spectra_nss.py) reproduces the dynesty pipeline's rest-frame magnitudes without
sedpy.

sedpy (Filter.obj_counts_hires, ab_mag): counts = trapz(lam T f_lam, lam) over the source
pixels where the interpolated transmission T > 0 (one extra pixel each side); the AB zero point
is the same integral of a flat 3631 Jy spectrum on the filter's own grid; m = -2.5 log10(counts /
zero). Here counts = f_lam @ weights, with weights the trapezoid coefficients times lam T.

Run in an environment with sedpy and the 'synthetic_*' filters (the prosp env; sedpy 0.4.1):
    python make_restframe_filters.py --meta /path/to/<library>/meta_data.pkl --out restframe_filters_c3k_hr.npz
It checks itself against sedpy.observate.getSED on random spectra and prints the worst difference.
"""
import argparse
import hashlib
import pickle

import numpy as np
import sedpy
from sedpy import observate

#: postprocess_parrot.py: uvj_names, gz_names, nuv_names (twomass_J appears in both)
FILTERS = ["bessell_U", "bessell_V", "twomass_J", "synthetic_u", "synthetic_g", "synthetic_i",
           "sdss_g0", "sdss_z0", "galex_NUV", "sdss_r0"]


def weights_for(f, wave):
    trans = np.interp(wave, f.wavelength, f.transmission, left=0.0, right=0.0)
    w = np.zeros_like(wave)
    pos = np.flatnonzero(trans > 0)
    if pos.size == 0:
        return w
    sl = slice(max(pos.min() - 1, 0), min(pos.max() + 2, wave.size))
    x = wave[sl]
    y = x * trans[sl]
    dx = np.diff(x)
    c = np.zeros_like(x)
    c[:-1] += 0.5 * dx
    c[1:] += 0.5 * dx
    w[sl] = c * y
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    meta = pickle.load(open(args.meta, "rb"))
    wave = np.asarray(meta["wavelengths"], float)
    flist = observate.load_filters(FILTERS)
    W = np.stack([weights_for(f, wave) for f in flist])
    zero = np.array([f.ab_zero_counts for f in flist])
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):                                   # random smooth positive spectra
        flam = np.exp(np.cumsum(rng.normal(0, 0.02, wave.size))) * 1e-18
        ref = observate.getSED(wave, flam, filterlist=flist)
        ours = -2.5 * np.log10((W @ flam) / zero)
        worst = max(worst, np.max(np.abs(ours - ref)))
    print(f"max |ours - sedpy getSED| over 20 random spectra: {worst:.2e} mag")
    np.savez_compressed(args.out, names=np.array(FILTERS), weights=W, zero_counts=zero,
                        wave_sha1=hashlib.sha1(wave.tobytes()).hexdigest(), n_wave=wave.size,
                        library=str(meta.get("stellar library", "")), sedpy_version=sedpy.__version__,
                        sedpy_filter_dir=observate.__file__)
    print(f"wrote {args.out}: {len(FILTERS)} filters on {wave.size} wavelengths "
          f"({wave.min():.0f}-{wave.max():.3g} A)")


if __name__ == "__main__":
    main()
