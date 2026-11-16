"""Faster E-FSPS emission lines when the line width is FIXED (our model: eline_sigma = 100 km/s).

E-FSPS (efsps/emulators/nebular.py, NebLineEmulator.predict_spec -> get_profiles) evaluates a
Gaussian for every (row, pixel, line): exp() on an N x 10,992 x 138 array. That is ~85% of
SpecModel's time per spectrum and most of its memory (~600 MB at 50 rows). The profiles depend
on the row only through eline_sigma, so with eline_sigma fixed the (pixel x line) profile
matrix is the same for every row: build it once from row 0 (E-FSPS's own get_profiles) and
take one matrix product with the line luminosities. Same numbers to float32 rounding
(test_spectra_nss.py checks against unpatched E-FSPS).

    import efsps_fast_lines
    efsps_fast_lines.enable(model)      # refuses if eline_sigma is free
    efsps_fast_lines.disable()

E-FSPS itself is not modified (never commit to efburnham/E-FSPS; a fork could take this).
"""
from efsps.emulators.nebular import NebLineEmulator
from efsps.sources.transforms import lsun_per_hz_to_maggies

_ORIGINAL = NebLineEmulator.predict_spec


def _predict_spec_fixed_sigma(self, theta):
    _ = self.predict_lines(theta)
    full = self.params
    n = full["eline_sigma"].shape[0]
    # row 0 of every per-row parameter: get_profiles reads eline_sigma only
    self.params = {k: (v[:1] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == n else v)
                   for k, v in full.items()}
    try:
        prof = self.get_profiles()[0]                         # (n_pix, n_lines)
    finally:
        self.params = full
    self.spec_lnu = self.line_luminosities @ prof.T            # (N, n_pix)
    self.spec = lsun_per_hz_to_maggies(self.spec_lnu, zred=self.params["zred"])
    return self.spec


def enable(model):
    """Patch NebLineEmulator for every model in this process; model is checked to have a
    fixed eline_sigma (the patch would be wrong otherwise)."""
    free = list(getattr(model, "free_parameter_order", []) or [])
    if "eline_sigma" in free:
        raise ValueError("efsps_fast_lines needs a FIXED eline_sigma; it is free in this model.")
    NebLineEmulator.predict_spec = _predict_spec_fixed_sigma


def disable():
    NebLineEmulator.predict_spec = _ORIGINAL
