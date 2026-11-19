"""Tests of the fit guards (nss_jax.py status field, iteration/time caps, NaN-safe termination,
bad-input handling), the provenance record (provenance.py) and the status handling of
make_catalog_nss.py.

 1. toys (run_nss_core directly, 5-d cube): log L = -inf everywhere and log L = NaN everywhere
    end with status "nonfinite" (standard start and prior start); a Gaussian toy ends "ok",
    with max_iterations=3 "capped", with max_seconds=0 "timed_out";
 2. nss_jax.py on a real galaxy with --max-iterations 3 / --max-seconds 1: status capped /
    timed_out, strict JSON (no NaN/Infinity tokens), no _post.h5, postprocessing recorded
    as skipped;
 3. an absent id and a galaxy with every flux NaN: status bad_input json without logZ,
    exit code 0; a catalogue file that does not exist: non-zero exit code and no json;
 4. real galaxies spanning redshift (small settings): status ok, provenance present and
    complete in the json and in the _post.h5; one line per galaxy;
 5. make_catalog_nss.py over those fits plus capped / timed-out / bad-input / non-finite /
    old-format jsons: every fit gets its row and its status, non-ok rows are masked,
    old files without status count as ok when they have logZ and did not time out, and the
    provenance reaches the HDF5 attrs and the FITS header.
Parts 2-5 use the private MINERVA subsets (git-ignored) when they are in nss/; without them,
parts 2-3 use the public uncover_test.csv and parts 4-5 are skipped (and say so).

    python test_fit_guards.py --out SCRATCH_DIR [--skip-real]
Ends RESULT: PASS or RESULT: FAIL.
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nss_jax import run_nss_core                                # noqa: E402

PY = sys.executable
SMALL = ["--n-live", "200", "--num-delete", "20", "--n-prior-start", "20000"]
#: (catalogue, id) spanning redshift: z_spec 0.12, 0.445, 2.09, 3.97, 6.12, 8.49
REAL = [("minerva_cosmos_dust21.fits", "2204022"), ("minerva_uds_test9.fits", "1174326"),
        ("minerva_uds_test9.fits", "1212906"), ("minerva_uds_test9.fits", "1197855"),
        ("minerva_uds_test9.fits", "1152437"), ("minerva_uds_dust27.fits", "1201575")]
PROV_REQUIRED = ("nss_commit", "nss_dirty", "nss_source", "efsps_commit", "emulator_md5",
                 "versions", "hostname", "cpu_model", "slurm", "created_utc", "catalog", "env")
LIBS = ("python", "jax", "jaxlib", "blackjax", "numpy", "scipy", "h5py", "astropy")

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        fails.append(name)


def strict_json(path):
    """Parse a json, refusing NaN / Infinity tokens (not valid JSON)."""
    def bad(c):
        raise ValueError(f"non-JSON token {c}")
    with open(path) as fh:
        return json.load(fh, parse_constant=bad)


def run(cmd, log):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    return r.returncode, time.time() - t0


def prov_problems(p, catalog=None):
    """What is missing or 'unknown' in a provenance record."""
    bad = [k for k in PROV_REQUIRED if k not in p]
    if bad:
        return bad
    if str(p["nss_commit"]).startswith("unknown") or p["nss_dirty"] is None:
        bad.append("nss_commit/dirty unknown")
    if p["nss_source"] not in ("git", "bundle"):
        bad.append(f"nss_source {p['nss_source']}")
    if p["efsps_commit"] == "unknown":
        bad.append("efsps_commit unknown")
    if not p.get("emulator_md5") or len(p["emulator_md5"]) != 32:
        bad.append("emulator_md5")
    bad += [f"version {k}" for k in LIBS if not p["versions"].get(k)]
    if not p["cpu_model"] or p["cpu_model"] == "unknown":
        bad.append("cpu_model")
    c = p["catalog"]
    if not all(k in c for k in ("path", "size_bytes", "mtime_utc")):
        bad.append("catalog path/size/mtime")
    elif catalog and os.path.abspath(catalog) != c["path"]:
        bad.append("catalog path differs")
    if not all(k in p["env"] for k in ("XLA_FLAGS", "NSS_EMU_DTYPE")):
        bad.append("env")
    return bad


# ------------------------------------------------------------------ 1. toys
def test_toys():
    print("1. toys (run_nss_core)")
    d = 5

    def neginf(u):
        return -jnp.inf + 0.0 * jnp.sum(u)

    def nan(u):
        return jnp.nan + 0.0 * jnp.sum(u)

    def gauss(u):
        return -0.5 * jnp.sum((u - 0.5) ** 2) / 0.1 ** 2

    kw = dict(n_live=50, num_delete=5, num_inner_steps=10, seed=0, log_every=10 ** 9,
              keep_discards=100)
    for name, ll, extra, want in (
            ("-inf, standard start", neginf, {}, "nonfinite"),
            ("-inf, prior start", neginf, {"n_prior_start": 500}, "nonfinite"),
            ("NaN, standard start", nan, {}, "nonfinite"),
            ("Gaussian", gauss, {}, "ok"),
            ("Gaussian, max_iterations 3", gauss, {"max_iterations": 3}, "capped"),
            ("Gaussian, max_seconds 0", gauss, {"max_seconds": 0.0}, "timed_out")):
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_nss_core(ll, d, **kw, **extra)
        it = res["iterations"]
        ok = res["status"] == want
        if want == "capped":
            ok &= it == 3
        if want == "timed_out":
            ok &= it == 1 and res["timed_out"]
        if want == "ok":
            ok &= bool(np.isfinite(np.mean(res["logz_draws"])))
        check(f"1 toy {name}", ok, f"status {res['status']} (want {want}), {it} iterations, "
              f"{time.time() - t0:.1f} s")


# ------------------------------------------------------------------ 2-3. CLI guards
def all_nan_copy(src, gid, dst):
    """One-row copy of a catalogue with every flux of gid set to NaN (header, units kept)."""
    if src.endswith(".csv"):
        lines = open(src).read().splitlines()
        head = lines[0].split(",")
        row = next(ln.split(",") for ln in lines[1:] if ln.split(",")[0] == gid)
        row = ["nan" if h.startswith("f_") else v for h, v in zip(head, row)]
        with open(dst, "w") as fh:
            fh.write(lines[0] + "\n" + ",".join(row) + "\n")
        return
    from astropy.io import fits
    with fits.open(src) as h:
        d = h[1].data
        sel = d[np.array([str(int(v)) for v in d["id"]]) == gid]
        for n in d.columns.names:
            if n.startswith("f_"):
                sel[n] = np.nan
        fits.BinTableHDU(data=sel, header=h[1].header).writeto(dst, overwrite=True)


def test_cli(out, cat, gid, nan_id):
    print(f"2-3. nss_jax.py guards on {os.path.basename(cat)} {gid}")
    base = [PY, "nss_jax.py", "--catalog", cat, "--out", out] + SMALL
    for tag, flags, want in (("capped", ["--max-iterations", "3"], "capped"),
                             ("timed", ["--max-seconds", "1"], "timed_out")):
        rc, wall = run(base + ["--id", gid, "--tag", tag] + flags, os.path.join(out, f"{gid}_{tag}.log"))
        js = os.path.join(out, f"{gid}_{tag}.json")
        try:
            s = strict_json(js)
            detail = (f"exit {rc}, status {s.get('status')}, {s.get('iterations')} iterations, "
                      f"timed_out {s.get('timed_out')}, postprocess {s.get('postprocess')}")
            ok = (rc == 0 and s.get("status") == want and "skipped" in (s.get("postprocess") or {})
                  and not os.path.exists(os.path.join(out, f"{gid}_{tag}_post.h5")))
            ok &= not prov_problems(s.get("provenance", {}), cat)
        except Exception as e:
            ok, detail = False, f"exit {rc}, json: {e!r}"
        check(f"2 {want} json, strict JSON, no _post.h5", ok, f"{detail}; {wall:.0f} s")

    absent = "999999999"
    rc, _ = run(base + ["--id", absent, "--tag", "absent"], os.path.join(out, "absent.log"))
    js = os.path.join(out, f"{absent}_absent.json")
    s = strict_json(js) if os.path.exists(js) else {}
    check("3a absent id -> bad_input, exit 0", rc == 0 and s.get("status") == "bad_input"
          and "logZ" not in s and s.get("id") == absent and not prov_problems(s.get("provenance", {}), cat)
          and {"reason", "tag", "catalog"} <= set(s), f"exit {rc}, {s.get('status')}: {s.get('reason')}")

    nan_cat = os.path.join(out, "all_nan" + (".csv" if cat.endswith(".csv") else ".fits"))
    all_nan_copy(cat, nan_id, nan_cat)
    rc, _ = run([PY, "nss_jax.py", "--catalog", nan_cat, "--out", out, "--id", nan_id,
                 "--tag", "allnan"] + SMALL, os.path.join(out, "allnan.log"))
    js = os.path.join(out, f"{nan_id}_allnan.json")
    s = strict_json(js) if os.path.exists(js) else {}
    check("3b every band NaN -> bad_input, exit 0", rc == 0 and s.get("status") == "bad_input"
          and "logZ" not in s, f"exit {rc}, {s.get('status')}: {s.get('reason')}")

    rc, _ = run(base[:2] + ["--catalog", os.path.join(out, "no_such_catalog.fits"), "--out", out,
                            "--id", gid, "--tag", "nocat"] + SMALL, os.path.join(out, "nocat.log"))
    check("3c missing catalogue file -> non-zero exit, no json", rc != 0 and not os.path.exists(
        os.path.join(out, f"{gid}_nocat.json")), f"exit {rc}")


# ------------------------------------------------------------------ 4. real galaxies
def test_real(out):
    import h5py
    print("4. real galaxies, small settings")
    rows = []
    for cat, gid in REAL:
        if not os.path.exists(os.path.join(HERE, cat)):
            print(f"  SKIP {gid}: {cat} not in nss/")
            continue
        rc, wall = run([PY, "nss_jax.py", "--catalog", cat, "--id", gid, "--tag", "guard",
                        "--out", out] + SMALL, os.path.join(out, f"{gid}_guard.log"))
        js = os.path.join(out, f"{gid}_guard.json")
        try:
            s = strict_json(js)
        except Exception as e:
            check(f"4 {gid}", False, f"exit {rc}: {e!r}")
            continue
        probs = prov_problems(s.get("provenance", {}), cat)
        post = os.path.join(out, f"{gid}_guard_post.h5")
        try:
            with h5py.File(post, "r") as f:
                hp = json.loads(f["fit"].attrs["provenance"])
                json.loads(f["fit"].attrs["postprocess_provenance"])
            post_ok = hp["nss_commit"] == s["provenance"]["nss_commit"]
        except Exception as e:
            post_ok, probs = False, probs + [f"_post.h5: {e!r}"]
        zs = (s.get("photometry") or {}).get("z_spec")
        rows.append((gid, zs, s.get("status"), s.get("iterations"), s.get("logZ"),
                     s.get("n_bands_used"), wall))
        check(f"4 {gid} status ok, provenance complete (json and _post.h5)",
              rc == 0 and s.get("status") == "ok" and not probs and post_ok
              and s.get("emu_dtype") == "float64" and s.get("peak_rss_mb", 0) > 0
              and s.get("process_elapsed_s", 0) > 0, f"problems {probs}" if probs else "")
    print("\n  id        z_spec  status  iterations   logZ     n_bands  wall s")
    for gid, zs, st, it, lz, nb, wall in rows:
        print(f"  {gid:8s}  {zs if zs is not None else float('nan'):6.3f}  {st:6s}  {it:10d}  "
              f"{lz:8.2f}  {nb:7d}  {wall:6.0f}")
    return rows


# ------------------------------------------------------------------ 5. make_catalog_nss
def test_catalog(out, guard_dir):
    import h5py
    from astropy.io import fits
    print("5. make_catalog_nss.py: statuses and provenance")
    cat = os.path.join(HERE, "minerva_uds_test9.fits")
    ind = os.path.join(out, "cat_in")
    shutil.rmtree(ind, ignore_errors=True)
    os.makedirs(ind)
    ok_ids = ["1174326", "1212906", "1197855", "1152437"]
    for gid in ok_ids:
        for suf in (".json", "_post.h5"):
            shutil.copy(os.path.join(guard_dir, f"{gid}_guard{suf}"), ind)
    # 1212906 as written before the status field: still "ok"
    s = strict_json(os.path.join(ind, "1212906_guard.json"))
    s.pop("status")
    json.dump(s, open(os.path.join(ind, "1212906_guard.json"), "w"), indent=1)
    # the capped and timed-out fits of part 2, renamed to other test9 ids
    tmpl = strict_json(os.path.join(guard_dir, "1197855_capped.json"))
    want = {gid: "ok" for gid in ok_ids}
    for gid, st, extra in (("1113001", "capped", {}),
                           ("1169924", "timed_out", {"timed_out": True}),
                           ("1198917", "nonfinite", {"logZ": None, "ess": None}),
                           ("1192179", "bad_input", None),
                           ("1241601", "timed_out", "old")):
        if extra is None:
            d = dict(strict_json(os.path.join(guard_dir, "1192179_allnan.json")), id=gid)
        elif extra == "old":                     # before the status field, timed out
            d = {k: v for k, v in tmpl.items() if k != "status"}
            d.update(id=gid, timed_out=True)
        else:
            d = dict(tmpl, id=gid, status=st, **extra)
        json.dump(d, open(os.path.join(ind, f"{gid}_guard.json"), "w"), indent=1)
        want[gid] = st
    cout = os.path.join(out, "cat_out")
    shutil.rmtree(cout, ignore_errors=True)
    rc, _ = run([PY, "make_catalog_nss.py", "--indir", ind, "--catalog", cat, "--out", cout],
                os.path.join(out, "make_catalog.log"))
    check("5a make_catalog_nss.py runs", rc == 0, f"exit {rc} (log {out}/make_catalog.log)")
    if rc != 0:
        return
    with h5py.File(os.path.join(cout, "quant_nss.h5"), "r") as f:
        codes = {int(k): v for k, v in json.loads(f["fit_status"].attrs["fit_status_codes"]).items()}
        got = {str(i): codes[int(c)] for i, c in zip(f["objid"][:], f["fit_status"][:])}
        okd = dict(zip(map(str, f["objid"][:]), f["ok"][:]))
        timed = dict(zip(map(str, f["objid"][:]), f["timed_out"][:]))
        prov = json.loads(f.attrs["provenance"])
    check("5b every fit has its row and status in quant_ h5", got == want,
          f"{got}" if got != want else f"{len(got)} fits")
    check("5c ok = 1 exactly for status ok", all((okd[g] == 1) == (want[g] == "ok") for g in want), "")
    check("5d timed_out dataset", timed["1169924"] == 1 and timed["1241601"] == 1
          and timed["1197855"] == 0, "")
    me = strict_json(os.path.join(guard_dir, "1197855_guard.json"))["provenance"]
    check("5e provenance in HDF5 attrs", list(prov["nss_commit"]) == [me["nss_commit"]]
          and list(prov["emulator_md5"]) == [me["emulator_md5"]], f"{prov['nss_commit']}")
    fcat = os.path.join(cout, "sps_catalog", "minerva_uds_test9_SPScatalog_spsv2.0.fits")
    with fits.open(fcat) as h:
        d, hd = h[1].data, h[1].header
        ids = [str(int(v)) for v in d["id"]]
        fs = dict(zip(ids, [x.strip() for x in d["fit_status"]]))
        z50 = dict(zip(ids, d["z_50"]))
        ess = dict(zip(ids, d["ess"]))
        nss_git, emu = hd.get("NSS_GIT"), hd.get("EMU_MD5")
        vjax = hd.get("V_JAX")
    check("5f FITS fit_status per galaxy", all(fs[g] == want[g] for g in want),
          str({g: fs[g] for g in want}))
    check("5g FITS: non-ok rows masked, ok rows filled",
          all(np.isnan(z50[g]) == (want[g] != "ok") for g in want)
          and all(np.isfinite(ess[g]) == (want[g] == "ok") for g in want), "")
    check("5h FITS header provenance", nss_git == me["nss_commit"] and emu == me["emulator_md5"]
          and vjax == me["versions"]["jax"], f"NSS_GIT {nss_git}, EMU_MD5 {emu}, V_JAX {vjax}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="scratch folder (default: a new temporary one)")
    p.add_argument("--skip-real", action="store_true", help="parts 1 and 2-3 only")
    args = p.parse_args()
    out = os.path.abspath(args.out or tempfile.mkdtemp(prefix="test_fit_guards_"))
    os.makedirs(out, exist_ok=True)
    print(f"outputs in {out}")
    test_toys()
    minerva = os.path.exists(os.path.join(HERE, "minerva_uds_test9.fits"))
    if minerva:
        test_cli(out, "minerva_uds_test9.fits", "1197855", "1192179")
    else:
        print("  (no MINERVA subsets in nss/: parts 2-3 on uncover_test.csv, 4-5 skipped)")
        test_cli(out, "uncover_test.csv", "58858", "58858")
    if minerva and not args.skip_real:
        test_real(out)
        test_catalog(out, out)
    print(f"RESULT: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")


if __name__ == "__main__":
    main()
