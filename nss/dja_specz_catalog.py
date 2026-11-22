"""DJA grade-3 spectroscopic redshifts attached to MINERVA sources by segmentation map
(PRODUCTION_PLAN task Z1b; first built 2026-11-17, rules revised through 2026-11-19).

This is the spec-z TRUTH for the Z1 photo-z benchmark (blanca/make_benchmark_sample.py reads
its dja_specz_all.csv). Photo-zs are NEVER used to decide anything here: eazy z_phot is only
copied into the by-eye change list for Wren to look at.

Rules (Wren's decisions; dates are when each was decided)
---------------------------------------------------------
Spectra (2026-11-17)
  - DJA nirspec_public_v4.5 (grade-3 z identical to v4.4 except rubies-egs52/28470,
    8.68 -> 6.38; checked). Grade 3 only, z > 0.
  - Spectra Wren rejects by eye (--veto, veto_spectra.csv, matched on `file`) are dropped first.
  - Spectrum position = the table's ra/dec (the APT planned position; msaexp extracts there
    without re-centring: profcen == 0 in the Zenodo v4.4 table). No astrometric offsets applied
    (DJA and MINERVA are both on Gaia via grizli); the per-programme median offset to the
    nearest MINERVA source within 0.3" is only reported (root_offsets_<field>.csv; medians
    <= 0.02" except excels-uds02, 0.16").
  - A spectrum is on a field if it falls on the segmentation map and either lands on a segment
    or has a catalogue source within 2".
Owner of each spectrum
  - Flux owner (2026-11-17, replacing the first version's seg-pixel rule): the F444W mosaic
    (clipped at 0) is summed per segment inside a disk of R_AP = 0.10" around the spectrum
    position (about half an open shutter's width, 0.20", plus the PSF); the owner is the
    segment with most light (owner_how = 'flux').
  - No mosaic light in the aperture (gaps; EGS: the v2.0 F444W mosaic on disk is zero over
    much of CEERS / gto-wide-egs1): the seg-map pixel at the position ('seg_pixel_noflux');
    on background: the nearest catalogue source within R_BG = 0.3" ('nearest_bg'); otherwise
    no owner ('none').
  - Centroid rule (2026-11-17, UDS 1079458): a catalogue centroid within R_CEN = 0.10" of the
    spectrum wins over the flux owner (APT targets are placed on centroids; a compact source
    sitting on a bigger neighbour's segment) ('centroid').
  - Far-owner guard (2026-11-18, "the slit is never more than 2" from the target"): owner
    centroid > FAR_MAX = 2" from the spectrum -> unassigned ('far_owner'; giant segments and
    sources missing from the catalogue; every one Wren checked was wrong).
  - By-eye overrides (--overrides, overrides.csv: file -> owner, -1 = no owner) ('by_eye').
  - use_phot = 0 owner (2026-11-19): the spectrum is left unassigned, NOT handed to a
    neighbour ('use_phot0'). So every source in the output has use_phot = 1.
  - Shreds (meeting 2026-11-18): one spec-z goes to ONE source only, the piece with most of the
    slit's light; no shred sharing, no SED-similarity rule.
Blends
  - possible_blend (2026-11-19, Wren: "a 20% cut is probably sufficient"; was any light on
    2026-11-18): another segment holds >= BLEND_FRAC = 20% of the aperture's F444W light.
    Every other segment with light is listed in blend_ids / blend_fracs regardless.
One entry per source
  - Several grade-3 spectra of one owner: the adopted z is the highest resolution (H, then M,
    then prism; then longest exposure). zspec_conflict = any of them differs from the adopted
    z by > ZTOL = 0.01 (1 + z) (the adopted z is still the highest resolution); dz_max,
    all_z list them. possible_blend = any of its spectra is; own_frac = the smallest owner
    share, second_frac the largest neighbour share over its spectra.

Change list (for Wren's by-eye check only; does not affect dja_specz_*.csv)
  Against the catalogue z_spec, with a catalogue z_spec traced to a DJA spectrum within
  R_OLD = 1.5" at the same z: 'moved' (the spectrum carrying the catalogue z is owned by a
  source without that z), 'extra_copy' (the owner already has the same catalogue z; the
  catalogue also gave it to a neighbour), 'z_changed', 'added', 'far_owner'. Wren's earlier
  verdicts (--verdicts, verdicts.csv) are attached and scored at the end of the log.

Inputs: the MINERVA catalogues, eazy zout, segmentation maps and F444W mosaics on DarkPhoenix
(paths in F below), the DJA table (--dja), and Wren's by-eye files (--veto, --overrides,
--verdicts; private). All outputs are PRIVATE (MINERVA data never goes in this public repo):
  <out>/dja_specz_<field>.csv and dja_specz_all.csv  (id, ra, dec, z_spec, field, grating,
       dja_root, dja_srcid, dja_file, n_spec, gratings, zspec_conflict, dz_max, all_z,
       possible_blend, blend_ids, own_frac, second_frac, method_changed, owner_how, spec_sep,
       use_phot, cat_z_spec)
  <out>/spectra_<field>.csv   every grade-3 spectrum on the field and its owner
  <out>/root_offsets_<field>.csv, changes.csv, changes_excel.csv, changes.html,
  <out>/dja_specz_catalog.log

  ~/envs/nss/bin/python dja_specz_catalog.py [--out DIR]     (~8 min, mostly reading F444W
                                                               mosaics; needs DarkPhoenix)
"""
import argparse
import os

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

D = "/Volumes/DarkPhoenix/Surveys/MINERVA/"
PRIV = "/Users/wren/Projects/minerva/nss_private_handoff/benchmark/specz_v45/"
DJA = "/Users/wren/Projects/minerva/dja_v4.4/nirspec_public_v4.5.csv"
F = {
    "UDS": dict(cat=D + "UDS/catalogs/n3.0_m3.1_v1.2.1/MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_mophongo.fits",
                zout=D + "UDS/catalogs/n3.0_m3.1_v1.2.1/MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits",
                seg=D + "UDS/catalogs/n3.0_m3.1_v1.2.1/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_SEGMAP.fits",
                img=D + "UDS/images/NIRCam/v3.0/uds-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_sci_f444w-matched.fits",
                url="https://minerva.colorado.edu/uds/latest/?ra={ra:.7f}&dec={dec:.7f}&zoom=10"),
    "COSMOS": dict(cat=D + "COSMOS/catalogs/n3.0_m3.0_v1.0.1/MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_Kf444w_SUPER_CATALOG_mophongo.fits",
                   zout=D + "COSMOS/catalogs/n3.0_v1.1/MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits",
                   seg=D + "COSMOS/catalogs/n3.0_v1.1/MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_SEGMAP.fits",
                   img=D + "COSMOS/images/NIRCam/v3.0/cosmos-grizli-v8.0-minerva-v3.0-40mas-f444w-clear_drc_sci_bkgsub.fits.gz",
                   url="https://minerva.colorado.edu/cosmos/?ra={ra:.7f}&dec={dec:.7f}&zoom=10"),
    "EGS": dict(cat=D + "EGS/catalogs/n2.0_m2.1_v1.3.1/MINERVA-EGS_n2.0_m2.1_v1.3.1_ACS+WEBB_Kf444w_SUPER_CATALOG_mophongo.fits",
                zout=D + "EGS/catalogs/n2.0_v1.3/MINERVA-EGS_n2.0_m2.1_v1.3.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits",
                seg=D + "EGS/catalogs/n2.0_v1.3/MINERVA-EGS_n2.0_v1.3_ACS+WEBB_SEGMAP.fits",
                img=D + "EGS/images/NIRCam/egs-grizli-v8.0-minerva-v2.0-40mas-f444w_drc_sci_f444w-matched.fits.gz",
                url="https://minerva.colorado.edu/egs/?ra={ra:.7f}&dec={dec:.7f}&zoom=10"),
}
R_BG = 0.3        # " aperture holds no segment flux -> nearest source within this
R_AP = 0.10       # " radius of the aperture at the spectrum position (about half an open shutter's
                  #   width, 0.20", plus the PSF); F444W flux in it is summed per segment
BLEND_FRAC = 0.20 # another segment with >= this share of the aperture's F444W light -> possible
                  #   blend (Wren 2026-11-19: "a 20% cut is probably sufficient"; was any light on
                  #   2026-11-18). Every segment with light is still listed in blend_ids/fracs
FAR_MAX = 2.0     # " owner centroid farther than this from the spectrum -> not assigned
                  #   (owner_how far_owner). Wren 2026-11-18: "the slit is never more than 2"
                  #   from the target"; every far owner on the 2026-11-17 list was wrong.
                  #   Shreds: the z goes to ONE piece only, the one with most of the slit's light
                  #   (decided at a meeting, 2026-11-18); no shred sharing.
R_CEN = 0.10      # " a catalogue centroid this close to the spectrum (APT targets are placed on
                  #   centroids) wins over the flux owner: compact sources on a bigger neighbour's
                  #   segment (UDS 1079458, Wren 2026-11-17)
R_OLD = 1.5       # " catalogue z_spec <-> DJA spectrum search radius (for the change list only)
ZTOL = 0.01       # |dz| / (1 + z)
S3 = "https://s3.amazonaws.com/msaexp-nirspec/extractions/"
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def res_rank(g):
    g = str(g).upper()
    return 0 if g.startswith("PRISM") else 2 if g.split("_")[0].endswith("H") else 1


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=PRIV, help="output folder (private)")
    ap.add_argument("--dja", default=DJA, help="DJA nirspec_public_v4.5 table")
    ap.add_argument("--veto", default=PRIV + "veto_spectra.csv", help="spectra Wren rejects")
    ap.add_argument("--overrides", default=PRIV + "overrides.csv",
                    help="Wren's by-eye owner per spectrum file (-1 = no owner)")
    ap.add_argument("--verdicts", default=PRIV + "verdicts.csv",
                    help="Wren's by-eye verdicts on earlier change lists")
    a = ap.parse_args()
    OUT = os.path.join(a.out, "")
    os.makedirs(OUT, exist_ok=True)
    LOG = open(OUT + "dja_specz_catalog.log", "w")

    dja = pd.read_csv(a.dja, low_memory=False)
    dja["grade"] = pd.to_numeric(dja.grade, errors="coerce")
    dja["z"] = pd.to_numeric(dja.z, errors="coerce")
    dja = dja[(dja.grade == 3) & (dja.z > 0)].reset_index(drop=True)
    veto = pd.read_csv(a.veto)
    log(f"veto list: {len(veto)} spectra; dropped {dja.file.isin(veto.file).sum()}")
    dja = dja[~dja.file.isin(veto.file)].reset_index(drop=True)
    dja["res"] = [res_rank(g) for g in dja.grating]
    log(f"DJA v4.5 grade-3 spectra with z > 0: {len(dja)}")

    cats, changes = [], []
    for field, P in F.items():
        log(f"\n===== {field}")
        c = fits.open(P["cat"], memmap=True)[1].data
        ids = np.asarray(c["id"]).astype(np.int64)
        ra, dec = np.asarray(c["ra"], float), np.asarray(c["dec"], float)
        zs = np.asarray(c["z_spec"], float)
        up = np.asarray(c["use_phot"]).astype(int)
        z = fits.open(P["zout"], memmap=True)[1].data
        assert np.array_equal(np.asarray(z["id"]).astype(np.int64), ids), "zout not row-matched"
        ez = np.asarray(z["z_phot"], float)      # copied into the change list only
        row_of = {int(i): k for k, i in enumerate(ids)}
        call = SkyCoord(ra * u.deg, dec * u.deg)

        sh = fits.open(P["seg"], memmap=True)
        hdu = sh[0] if sh[0].data is not None else sh[1]
        seg, wcs = hdu.data, WCS(hdu.header)
        ny, nx = seg.shape
        pix = proj_plane_pixel_scales(wcs)[0] * 3600

        def to_pix(r, d):
            x, y = wcs.all_world2pix(r, d, 0)
            return np.round(x).astype(int), np.round(y).astype(int)

        t = np.random.default_rng(1).choice(len(ids), 5000, replace=False)
        x, y = to_pix(ra[t], dec[t])
        ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        same = np.asarray([seg[b, a_] for a_, b in zip(x[ok], y[ok])]) == ids[t][ok]
        log(f"seg map {hdu.data.shape}, {pix:.3f}\"/pix; segment id at the catalogue centroid == "
            f"catalogue id for {same.mean() * 100:.2f}% of 5000 random sources")

        # spectra on this map
        x, y = to_pix(dja.ra.values, dja.dec.values)
        on = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        S = dja[on].copy()
        S["x"], S["y"] = x[on], y[on]
        S["seg_id"] = [int(seg[b, a_]) for a_, b in zip(S.x, S.y)]
        cs = SkyCoord(S.ra.values * u.deg, S.dec.values * u.deg)
        nn, nsep, _ = cs.match_to_catalog_sky(call)
        S["near_id"], S["near_sep"] = ids[nn], nsep.arcsec
        # off the image footprint (seg 0 and nothing within 2") is not MINERVA's
        S = S[(S.seg_id > 0) | (S.near_sep < 2.0)].reset_index(drop=True)
        cs = SkyCoord(S.ra.values * u.deg, S.dec.values * u.deg)
        log(f"grade-3 spectra on the {field} map: {len(S)}")

        # programme offsets, reported only (not applied)
        m = S.near_sep < 0.3
        j = np.array([row_of[i] for i in S.near_id[m]])
        S.loc[m, "dra"] = (ra[j] - S.ra[m]) * np.cos(np.deg2rad(dec[j])) * 3600
        S.loc[m, "ddec"] = (dec[j] - S.dec[m]) * 3600
        off = S[m].groupby("root").agg(n=("file", "size"), dra=("dra", "median"), ddec=("ddec", "median"))
        off = off[off.n >= 8]
        off["tot"] = np.hypot(off.dra, off.ddec)
        log(f"[info, not applied] median offset spectrum -> nearest MINERVA source within 0.3\" per "
            f"programme root (>= 8 spectra, N {len(off)} roots): median {off.tot.median():.3f}\", "
            f"max {off.tot.max():.3f}\" ({off.tot.idxmax()})")
        off.sort_values("tot", ascending=False).to_csv(OUT + f"root_offsets_{field.lower()}.csv")

        # pixel owner (the first version's rule), kept for comparison and as the no-flux fallback
        S["pix_owner"] = np.where(S.seg_id > 0, S.seg_id, np.where(S.near_sep < R_BG, S.near_id, -1))

        # owner by F444W flux: sum the image flux (clipped at 0) per segment inside R_AP
        log(f"reading {P['img'].split('/')[-1]} ...")
        img = fits.getdata(P["img"])
        assert img.shape == seg.shape, "F444W image and seg map on different grids"
        r = int(np.ceil(R_AP / pix))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) * pix ** 2 <= R_AP ** 2
        own, frac, shares = [], [], []
        for a_, b in zip(S.x, S.y):
            y0, x0 = max(b - r, 0), max(a_ - r, 0)
            sg = np.asarray(seg[y0:b + r + 1, x0:a_ + r + 1])
            im = np.clip(np.nan_to_num(np.asarray(img[y0:b + r + 1, x0:a_ + r + 1], float)), 0, None)
            dk = disk[r - (b - y0):r - (b - y0) + sg.shape[0], r - (a_ - x0):r - (a_ - x0) + sg.shape[1]]
            sv, fv = sg[dk], im[dk]
            keep = sv > 0
            if not keep.any() or fv[keep].sum() <= 0:
                own.append(-1); frac.append(np.nan); shares.append({})
                continue
            u_, inv = np.unique(sv[keep], return_inverse=True)
            fl = np.bincount(inv, weights=fv[keep])
            fr = fl / fl.sum()
            o = np.argsort(-fr)
            own.append(int(u_[o[0]])); frac.append(float(fr[o[0]]))
            shares.append({int(u_[k]): float(fr[k]) for k in o if fr[k] > 0})
        del img
        S["flux_owner"], S["own_frac"] = own, frac
        # no image flux in the aperture (gaps in the F444W mosaic): fall back to the seg pixel
        S["owner"] = np.where(S.flux_owner > 0, S.flux_owner, S.pix_owner)
        S["owner_how"] = np.where(S.flux_owner > 0, "flux",
                                  np.where(S.seg_id > 0, "seg_pixel_noflux",
                                           np.where(S.owner > 0, "nearest_bg", "none")))
        # a catalogue centroid within R_CEN wins
        cen = (S.near_sep.values < R_CEN) & (S.near_id.values != S.owner.values)
        S.loc[cen, "owner"] = S.near_id[cen]
        S.loc[cen, "owner_how"] = "centroid"
        S["method_changed"] = (S.owner != S.pix_owner)
        S["owner_sep"] = [call[row_of[o]].separation(cs[k]).arcsec if o > 0 else np.nan
                          for k, o in enumerate(S.owner)]
        # far-owner guard: the slit is never more than FAR_MAX from its target
        S["far_owner"] = S.owner_sep.values > FAR_MAX
        S["far_owner_id"] = np.where(S.far_owner, S.owner, -1)
        S.loc[S.far_owner, "owner_how"] = "far_owner"
        S.loc[S.far_owner, "owner"] = -1
        # by-eye overrides (Wren), per spectrum file
        ov = pd.read_csv(a.overrides) if os.path.exists(a.overrides) else \
            pd.DataFrame(columns=["file", "owner"])
        ovm = dict(zip(ov.file, ov.owner.astype(int)))
        hit = S.file.isin(ovm).values
        if hit.any():
            S.loc[hit, "owner"] = [ovm[f] for f in S.file[hit]]
            S.loc[hit, "owner_how"] = "by_eye"
        log(f"by-eye overrides applied: {hit.sum()} spectra")
        # no spec-z on use_phot = 0 sources (Wren 2026-11-19): such a spectrum is left unassigned,
        # not handed to a neighbour
        up0 = np.array([o > 0 and up[row_of[int(o)]] != 1 for o in S.owner])
        S["use_phot0_id"] = np.where(up0, S.owner, -1)
        S.loc[up0, "owner_how"] = "use_phot0"
        S.loc[up0, "owner"] = -1
        log(f"owner has use_phot = 0 -> unassigned: {up0.sum()} spectra "
            f"({len(set(S.use_phot0_id[up0]))} sources)")
        # shares of the slit light, reported for the FINAL owner (centroid / by-eye can differ from
        # the flux owner); every other segment with light is listed and makes a possible blend
        of, sid, sf, li, lf = [], [], [], [], []
        for shr, o in zip(shares, S.owner):
            oth = sorted(((v, k) for k, v in shr.items() if k != o), reverse=True)
            of.append(shr.get(int(o), 0.0) if shr else np.nan)
            sid.append(oth[0][1] if oth else -1)
            sf.append(oth[0][0] if oth else 0.0)
            li.append(" ".join(str(k) for v, k in oth))
            lf.append(" ".join(f"{v:.2f}" if v >= 0.01 else "<0.01" for v, k in oth))
        S["own_frac"], S["second_id"], S["second_frac"], S["blend_ids"], S["blend_fracs"] = of, sid, sf, li, lf
        S["possible_blend"] = (S.second_frac.fillna(0) >= BLEND_FRAC) & (S.owner > 0)
        log("owners: " + ", ".join(f"{k} {v}" for k, v in S.owner_how.value_counts().items())
            + f"; flux owner differs from the pixel owner: {S.method_changed.sum()}; possible blends "
              f"(another segment >= {BLEND_FRAC:.0%} of the aperture light): {S.possible_blend.sum()} of {len(S)} "
              f"(other segment's share >= 5%: {((S.second_frac >= 0.05) & (S.owner > 0)).sum()}, "
              f">= 20%: {((S.second_frac >= 0.2) & (S.owner > 0)).sum()})")
        S.drop(columns=["thumb", "fnu", "flam", "FITS", "flam_v3"], errors="ignore").to_csv(
            OUT + f"spectra_{field.lower()}.csv", index=False)

        # one entry per owner
        rows = []
        for o, g in S[S.owner > 0].groupby("owner"):
            g = g.sort_values(["res", "exptime"], ascending=[False, False])
            best = g.iloc[0]
            dzs = np.abs(g.z.values - best.z) / (1 + best.z)
            k = row_of[int(o)]
            rows.append(dict(
                id=int(o), ra=ra[k], dec=dec[k], z_spec=best.z, field=field,
                grating=best.grating, dja_root=best.root, dja_srcid=best.srcid, dja_file=best.file,
                n_spec=len(g), gratings=" ".join(sorted(set(g.grating))),
                zspec_conflict=bool((dzs > ZTOL).any()), dz_max=float(dzs.max()),
                all_z=" ".join(f"{v:.4f}" for v in g.z),
                possible_blend=bool(g.possible_blend.any()),
                blend_ids=" ".join(sorted(set(" ".join(g.blend_ids).split()))),
                own_frac=float(np.nanmin(g.own_frac)), second_frac=float(np.nanmax(g.second_frac)),
                method_changed=bool(g.method_changed.any()),
                owner_how=",".join(sorted(set(g.owner_how))),
                spec_sep=float(np.nanmin(g.owner_sep)), use_phot=int(up[k]),
                cat_z_spec=zs[k] if zs[k] > 0 else np.nan))
        Cf = pd.DataFrame(rows)
        log(f"MINERVA sources with a DJA grade-3 z: {len(Cf)}; conflicts (> {ZTOL}(1+z) between "
            f"spectra) {Cf.zspec_conflict.sum()}; possible blends {Cf.possible_blend.sum()}; "
            f"z >= 7: {(Cf.z_spec >= 7).sum()}, z >= 9: {(Cf.z_spec >= 9).sum()}")
        Cf.to_csv(OUT + f"dja_specz_{field.lower()}.csv", index=False)
        cats.append(Cf)

        # ---- comparison with the catalogue's z_spec (for the by-eye list only)
        new = {int(i): rr for i, rr in zip(Cf.id, Cf.itertuples())}
        agree = np.zeros(len(ids), bool)
        for i, rr in new.items():
            k = row_of[i]
            agree[k] = zs[k] > 0 and abs(zs[k] - rr.z_spec) / (1 + rr.z_spec) < ZTOL
        hz = np.flatnonzero(zs > 0)
        # catalogue spec-z equal to a DJA grade-3 z of a spectrum within R_OLD (where it likely came from)
        a1, b1, sp, _ = cs.search_around_sky(call[hz], R_OLD * u.arcsec)
        pr = pd.DataFrame(dict(k=hz[a1], s=b1, sep=sp.arcsec))
        pr = pr[np.abs(S.z.values[pr.s] - zs[pr.k]) / (1 + S.z.values[pr.s]) < ZTOL]
        from_dja = set(pr.k)
        exact = pd.Series(np.abs(S.z.values[pr.s] - zs[pr.k]) < 1e-4).groupby(pr.k.values).any()
        log(f"catalogue z_spec > 0: {len(hz)}; equal to a DJA v4.5 grade-3 z within {R_OLD}\" "
            f"(to 0.01(1+z)): {len(from_dja)} (to 1e-4: {int(exact.sum())}); not DJA: {len(hz) - len(from_dja)}")
        moved_new = set()
        for k in sorted(from_dja):
            if agree[k]:
                continue
            p = pr[pr.k == k].sort_values("sep")
            # the spectrum carrying this z is owned by someone else
            sidx = p.s.iloc[0]
            o = int(S.owner.iat[sidx])
            if o <= 0:
                continue                     # unassigned (far owner / none): listed below
            if o == ids[k]:
                continue                     # own spectrum but merged z differs: shows as z_changed
            dup = zs[row_of[o]] > 0 and abs(zs[row_of[o]] - zs[k]) / (1 + zs[k]) < ZTOL if o > 0 else False
            changes.append(dict(field=field, kind="extra_copy" if dup else "moved", old_id=int(ids[k]),
                                new_id=o, s=S.iloc[sidx], old_zspec_cat=zs[k],
                                cat_z_exact=bool(abs(S.z.iat[sidx] - zs[k]) < 1e-4)))
            moved_new.add(o)
        for i, rr in new.items():
            k = row_of[i]
            if agree[k] or i in moved_new:
                continue
            sidx = S.index[S.file == rr.dja_file][0]
            kind = "z_changed" if zs[k] > 0 else "added"
            changes.append(dict(field=field, kind=kind, old_id=i if zs[k] > 0 else -1, new_id=i,
                                s=S.loc[sidx], old_zspec_cat=zs[k] if zs[k] > 0 else np.nan))
        for sidx in np.flatnonzero(S.far_owner.values):
            fo = int(S.far_owner_id.iat[sidx])
            changes.append(dict(field=field, kind="far_owner", old_id=fo if zs[row_of[fo]] > 0 else -1,
                                new_id=fo, s=S.iloc[sidx],
                                old_zspec_cat=zs[row_of[fo]] if zs[row_of[fo]] > 0 else np.nan))
        for d in changes:
            if "fitsmap" in d or d["field"] != field:
                continue
            s = d.pop("s")
            o, n = d["old_id"], d["new_id"]
            ko, kn = row_of.get(o), row_of[n]
            sc = SkyCoord(s.ra * u.deg, s.dec * u.deg)
            nr = new.get(n)
            d.update(dja_root=s.root, dja_srcid=s.srcid, dja_file=s.file, grating=s.grating,
                     dja_z=nr.z_spec if (nr is not None and d["kind"] != "far_owner") else s.z,
                     old_eazy=ez[ko] if ko is not None else np.nan, new_eazy=ez[kn],
                     old_sep=call[ko].separation(sc).arcsec if ko is not None else np.nan,
                     new_sep=call[kn].separation(sc).arcsec,
                     new_zspec_cat=zs[kn] if zs[kn] > 0 else np.nan,
                     possible_blend=bool(s.possible_blend), blend_ids=s.blend_ids,
                     blend_fracs=s.blend_fracs, own_frac=s.own_frac, second_frac=s.second_frac,
                     pixel_owner=int(s.pix_owner), method_changed=bool(s.method_changed),
                     owner_how=s.owner_how, conflict=bool(nr.zspec_conflict) if nr is not None else False,
                     fitsmap=P["url"].format(ra=s.ra, dec=s.dec),
                     spec_png=S3 + s.root + "/" + s.file.split("spec")[0] + "fnu.png",
                     slit_thumb=f"https://grizli-cutout.herokuapp.com/thumb?size=1.5&scl=4.0&invert=True"
                                f"&filters=f444w-clear&pl=2&coord={s.ra:.6f}%2C{s.dec:.6f}&nirspec=True"
                                f"&dpi_scale=6&nrs_lw=0.5&nrs_alpha=0.8&metafile={str(s.msamet)[:13]}")

    A = pd.concat(cats, ignore_index=True)
    A.to_csv(OUT + "dja_specz_all.csv", index=False)
    C = pd.DataFrame(changes)
    order = {"moved": 0, "far_owner": 1, "z_changed": 2, "extra_copy": 3, "added": 4}
    C["o"] = C.kind.map(order)
    C = C.sort_values(["o", "dja_z"], ascending=[True, False]).drop(columns="o")
    # Wren's earlier verdicts
    V = pd.read_csv(a.verdicts)
    vk = {(r.field, int(r.old_id), int(r.new_id)): r for r in V.itertuples()}
    C["prev_verdict"] = [vk[(f, int(o), int(n))].verdict if (f, int(o), int(n)) in vk else ""
                         for f, o, n in zip(C.field, C.old_id, C.new_id)]
    C["prev_note"] = [vk[(f, int(o), int(n))].note if (f, int(o), int(n)) in vk else ""
                      for f, o, n in zip(C.field, C.old_id, C.new_id)]
    # by-eye set: what the flux rule changed, every far owner, and z >= 7 not yet judged
    # (far owners are settled by the 2" rule, Wren 2026-11-18: listed, not checked)
    C["check"] = ((C.method_changed | (C.dja_z >= 7)) & (C.kind != "far_owner")
                  & (C.prev_verdict == "") & (C.kind != "added")) | (C.method_changed & (C.kind == "added"))
    C["by_eye"] = np.where(C.check, "yes", "no")
    C["z_spec"] = C.dja_z
    first = ["field", "kind", "by_eye", "old_id", "new_id", "z_spec", "old_eazy", "new_eazy", "old_sep",
             "new_sep", "blend_ids", "blend_fracs", "own_frac", "pixel_owner", "method_changed",
             "prev_verdict", "prev_note"]
    rest = [c for c in C.columns if c not in first + ["check", "dja_z", "fitsmap"]]
    X = C[first + rest + ["fitsmap"]].copy()
    X["blend_ids"] = [" ".join(str(int(float(t))) for t in str(v).split()) if isinstance(v, str) or v == v else ""
                      for v in X.blend_ids]
    for c in ("old_eazy", "new_eazy", "z_spec"):
        X[c] = X[c].round(3)
    for c in ("old_sep", "new_sep"):
        X[c] = X[c].round(3)
    X.to_csv(OUT + "changes.csv", index=False)
    # Excel version: links as HYPERLINK formulas (clickable when the csv is opened in Excel)
    Y = X.copy()
    for c, lab in (("spec_png", "spectrum"), ("slit_thumb", "slit"), ("fitsmap", "FITSMAP")):
        Y[c] = [f'=HYPERLINK("{v}","{lab}")' for v in Y[c]]
    Y["blend_ids"] = ['="' + v + '"' if v else "" for v in Y.blend_ids]   # keep ids as text
    Y["blend_fracs"] = ['="' + v + '"' if isinstance(v, str) and v else "" for v in Y.blend_fracs]
    Y.to_csv(OUT + "changes_excel.csv", index=False)
    log("\n" + C.groupby(["kind", "field"]).size().to_string())
    log(f"by-eye set (flux rule changed the owner or z >= 7, not yet judged; far owners not checked): {C.check.sum()}")
    # score against Wren's verdicts: which ids get the spec-z now
    have = {(r.field, int(r.id)): r.z_spec for r in A.itertuples()}
    log("\nWren's verdicts vs the flux rule (ids that should carry the z / ids that do):")
    for r in V.itertuples():
        if r.verdict == "superseded":  # a later verdict on the same source replaces it
            continue
        want = [int(float(x)) for x in str(r.right_ids).split()] if str(r.right_ids) not in ("", "nan") else []
        cand = ({int(r.old_id), int(r.new_id)} | set(want)) - {-1}
        got = [i for i in sorted(cand) if (r.field, i) in have and abs(have[(r.field, i)] - r.z_spec) / (1 + r.z_spec) < 0.02]
        if r.verdict == "shred":       # meeting 2026-11-18: one piece only (most slit light)
            ok = len(got) == 1 and got[0] in want
        else:
            ok = set(got) == set(want)
        log(f"  {r.field:6s} {r.old_id}->{r.new_id} {r.verdict:14s} want {want if r.verdict != 'shred' else 'one of ' + str(want)} got {got} -> "
            f"{'OK' if ok else 'WRONG'}")

    def fmt(v, c):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3f}" if "sep" in c else f"{v:.2f}"
        return str(v)

    cols = ["check", "field", "kind", "map", "old_id", "new_id", "old_eazy", "new_eazy", "dja_z",
            "old_zspec_cat", "new_zspec_cat", "dja_root", "dja_srcid", "grating", "old_sep", "new_sep",
            "possible_blend", "blend_ids", "conflict", "owner_how", "cat_z_exact", "links"]
    body = []
    for _, r in C.iterrows():
        cells = []
        for c in cols:
            if c == "map":
                v = f'<a href="{r.fitsmap}" target="_blank">FITSMAP</a>'
            elif c == "links":
                v = (f'<a href="{r.spec_png}" target="_blank">spectrum</a> '
                     f'<a href="{r.slit_thumb}" target="_blank">slit</a>')
            else:
                v = fmt(r[c], c)
            cells.append(f"<td>{v}</td>")
        body.append(("<tr class=chk>" if r.check else "<tr>") + "".join(cells) + "</tr>")
    open(OUT + "changes.html", "w").write(
        "<!doctype html><meta charset=utf-8><title>DJA spec-z changes</title><style>"
        "body{font:13px system-ui;margin:16px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:2px 6px;white-space:nowrap}th{background:#eee;position:sticky;top:0}"
        "tr.chk td{background:#fff7d6}</style>"
        "<p><b>moved</b>: the catalogue z_spec sits on old_id, but the DJA spectrum with that z lies on "
        "new_id's segment (new_id has no catalogue z_spec). <b>extra_copy</b>: the catalogue gives the same z_spec to "
        "new_id (right) and to its neighbour old_id (wrong). <b>z_changed</b>: the source's own DJA z differs from its catalogue z_spec by "
        "&gt; 0.01(1+z). <b>added</b>: DJA grade-3 z for a source with no catalogue z_spec. Separations "
        "(\") from the DJA (APT) position. Photo-zs are shown for information only; they were not used. "
        "Yellow rows: by-eye set (z &ge; 7 plus a random 50 moved/z_changed/extra_copy).</p><table><tr>"
        + "".join(f"<th>{c}</th>" for c in cols) + "</tr>" + "".join(body) + "</table>")
    LOG.close()


if __name__ == "__main__":
    main()
