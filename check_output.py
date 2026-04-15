import numpy as np
import matplotlib.pyplot as plt
import glob, os, time, random, sys
import astropy.io.fits as fits
import h5py


def plot_all(objid, plt_jy=True, sdir=None, imhead=None, show=None, add_text=None):
    objid_arr_sed = fsed["objid"][:]
    if objid not in objid_arr_sed:
        print(f'no sed for obj {objid}!')
        return None

    idx_sed = np.where(objid_arr_sed == objid)[0][0]

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0)
    resid = fig.add_axes([0.1, 0.1, 0.4, 0.25])
    phot = fig.add_axes([0.1, 0.35, 0.4, 0.5])

    # --------------------
    # SED
    # --------------------
    zred = fsed["zred"][idx_sed]
    print("zml:", zred)

    obsmags = fsed["obsmag"][idx_sed].astype(float)
    obsunc = fsed["obsmag_unc"][idx_sed].astype(float)
    modmags = fsed["modmag_map"][idx_sed].astype(float)
    modspec = fsed["modspec_map"][idx_sed].astype(float)

    weff = fsed["weff"][:].astype(float) / 1e4
    wavspec = fsed["wavspec"][:].astype(float) * (1 + zred) / 1e4

    mask = np.isfinite(obsmags)
    obsmags = obsmags[mask]
    obsunc = obsunc[mask]
    weff_use = weff[mask]
    modmags = modmags[mask]

    # same clipping as before
    obsunc = np.clip(obsunc, a_min=np.abs(obsmags) * 0.05, a_max=None)

    if plt_jy:
        obsmags *= 3631.0
        obsunc *= 3631.0
        modmags *= 3631.0
        modspec *= 3631.0

    phot.errorbar(
        weff_use, modmags, fmt='o', color='firebrick',
        label='model photometry', zorder=100,
        elinewidth=1, mec='k', mew=0.2
    )
    phot.errorbar(
        weff_use, obsmags, yerr=obsunc, color='black',
        fmt='o', label='observed photometry', zorder=101
    )
    phot.plot(
        wavspec, modspec, '-', color='firebrick',
        label='model spectrum', zorder=-100
    )

    xmin, xmax = weff_use.min() * 0.8, weff_use[:-1].max() * 1.5
    phot.set_xlim(xmin, xmax)

    pos_obs = obsmags[np.isfinite(obsmags) & (obsmags > 0)]
    ymin = np.nanmin(pos_obs) * 0.3 if len(pos_obs) > 0 else 1e-6
    ymax = np.nanmax(obsmags[np.isfinite(obsmags)]) * 2 if np.isfinite(obsmags).any() else 1.0
    phot.set_ylim(ymin, ymax)
    resid.set_xlim(xmin, xmax)

    if (obsmags < 0).any():
        y0 = 10 ** ((np.log10(ymax) - np.log10(ymin)) / 20.0) * ymin
        for x0 in weff_use[obsmags < 0]:
            phot.plot(
                x0, y0, linestyle='none', marker=u'$\u2193$',
                markersize=16, mew=0.5, mec='k', color='k'
            )

    photchi = (modmags - obsmags) / obsunc
    resid.plot(weff_use, photchi, 'o', color='firebrick')
    resid.axhline(0.0, linestyle=':', color='grey')

    y1, y2 = resid.get_ylim()
    yl = np.nanmax([np.abs(y1), np.abs(y2)])
    resid.set_ylim(-yl, yl)

    phot.text(
        0.97, 0.92, str(objid), fontsize=18, transform=phot.transAxes,
        ha='right', va="top", weight='bold'
    )
    if add_text is not None:
        phot.text(
            0.97, 0.82, add_text, fontsize=18, transform=phot.transAxes,
            ha='right', va="top"
        )

    lblsize = 16
    resid.set_xlabel(r'$\lambda_{\rm{observed}} \; [\mu m]$', fontsize=lblsize)
    resid.set_ylabel(r'$\chi$', fontsize=lblsize)
    ylab = 'Flux density [maggies]'
    if plt_jy:
        ylab = 'Flux density [Jy]'
    phot.set_ylabel(ylab, fontsize=lblsize)
    phot.set_yscale('log', nonpositive='clip')
    phot.set_xscale('log', nonpositive='clip')
    resid.set_xscale('log', nonpositive='clip', subs=(1, 2, 3, 4, 5, 6, 8))
    resid.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    resid.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    phot.set_xticklabels([])
    fig.tight_layout()

    # --------------------
    # SFH
    # --------------------
    objid_arr_sfh = fsfh["objid"][:]
    if objid not in objid_arr_sfh:
        print(f'no sfh for obj {objid}!')
        return fig

    idx_sfh = np.where(objid_arr_sfh == objid)[0][0]
    sfh = fsfh["sfh"][idx_sfh]
    i50 = sfh.shape[0] // 2
    tbins = sfh.shape[1]
    agebins_max = fsfh["agebins_max"][idx_sfh]
    age = np.logspace(1, agebins_max, tbins) / 1e9

    axsfh = fig.add_axes([0.6, 0.1, 0.4, 0.75])
    axsfh.step(age, sfh[i50, :], color='k')
    if i50 - 1 >= 0 and i50 + 1 < sfh.shape[0]:
        axsfh.fill_between(age, sfh[i50 - 1, :], sfh[i50 + 1, :],
                           alpha=0.5, color='gray', step='pre')
    if i50 - 2 >= 0 and i50 + 2 < sfh.shape[0]:
        axsfh.fill_between(age, sfh[i50 - 2, :], sfh[i50 + 2, :],
                           alpha=0.2, color='gray', step='pre')

    axsfh.set_xlim(1e-3, np.max(age))
    axsfh.set_xscale('log')
    fs = 18
    axsfh.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    axsfh.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]', fontsize=fs)

    # --------------------
    # redshift histogram
    # --------------------
    objid_arr_chain = fchain["objid"][:]
    if objid in objid_arr_chain:
        idx_chain = np.where(objid_arr_chain == objid)[0][0]

        theta_labels_raw = fchain["theta_labels"][:]
        theta_labels = [
            t.decode() if isinstance(t, (bytes, np.bytes_)) else str(t)
            for t in theta_labels_raw
        ]
        thetaidx_in_z = dict(zip(theta_labels, np.arange(len(theta_labels))))

        _chain = fchain["chains"][idx_chain]
        _chain_for_z = _chain[:, thetaidx_in_z["zred"]]

        axz = fig.add_axes([0.12, 0.65, 0.12, 0.15])
        axz.set_yticks([])
        axz.hist(_chain_for_z, color="black", density=True)

        del _chain, _chain_for_z

    # --------------------
    # image
    # --------------------
    if objid not in id_mega:
        print(f'no image for obj {objid}!')
    else:
        crossid_here = np.where(id_mega == float(objid))[0][0]
        ra_here = ra_mega[crossid_here]
        dec_here = dec_mega[crossid_here]

        if sfh[i50, 0] < 0.5 * np.nanmax(sfh[min(i50 + 2, sfh.shape[0] - 1), :]) * 1.1:
            thumb = fig.add_axes([0.62, 0.5, 0.12, 0.3])
        else:
            thumb = fig.add_axes([0.62, 0.15, 0.12, 0.3])

        y_pix, x_pix = wcs_jwst.wcs_world2pix(ra_here, dec_here, 0)
        data_around = data[
            int(x_pix - 10 / pixscale):int(x_pix + 10 / pixscale),
            int(y_pix - 10 / pixscale):int(y_pix + 10 / pixscale)
        ]
        data_show = data[
            int(x_pix - 2 / pixscale):int(x_pix + 2 / pixscale),
            int(y_pix - 2 / pixscale):int(y_pix + 2 / pixscale)
        ]
        imsize = np.shape(data_show)[0]
        mean_pix, median_pix, stddev_pix = sigma_clipped_stats(data_around)

        if np.isnan(stddev_pix):
            for _ in range(10):
                std = np.nanstd(data_around)
                data[np.where(data > 3 * std)] = np.nan
            stddev_pix = std

        thumb.set_xticks([])
        thumb.set_yticks([])
        thumb.imshow(data_show, vmin=-2 * stddev_pix, vmax=10 * stddev_pix, cmap="viridis")

        clip = imsize / 2 - 0.5
        color1 = 'red'
        thumb.plot([clip, clip], [clip * 0.7, clip * 0.2], color=color1, lw=0.5)
        thumb.plot([clip, clip], [clip * 1.3, clip * 1.8], color=color1, lw=0.5)
        thumb.plot([clip * 0.7, clip * 0.2], [clip, clip], color=color1, lw=0.5)
        thumb.plot([clip * 1.3, clip * 1.8], [clip, clip], color=color1, lw=0.5)

        del data_around, data_show

    if sdir is not None:
        plt.savefig(sdir + f'full_{objid}_{imhead}.png', bbox_inches='tight')
    if show is None:
        plt.ioff()

    return fig

cat_dir = "/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/"
photozcat = fits.open(cat_dir+"sps_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG_SPScatalog_spsv1.0.fits")
photozcat_data = photozcat[1].data
photozcat_header = photozcat[1].header

id_mega_photo = photozcat_data['id']
nbands_mega = photozcat_data['nbands']
zspec_mega = photozcat_data['z_spec']
zbest_phot = photozcat_data['z_ml']
z_phot = photozcat_data['z_50']
el_z_phot = z_phot-photozcat_data['z_16']
eh_z_phot = photozcat_data['z_84']-z_phot

ffsed = cat_dir+'spec_phisfh.h5'
ffsfh = cat_dir+'sfh_phisfh.h5'
ffchain = cat_dir+'chains_phisfh.h5'
fsed = h5py.File(ffsed, "r")
fsfh = h5py.File(ffsfh, "r")
fchain = h5py.File(ffchain, "r")


print(z_phot[np.where(id_mega_photo==1235366)])
print(z_phot[np.where(id_mega_photo==1235367)])

plot_all(1235366,sdir="./")
