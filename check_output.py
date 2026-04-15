import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import glob, os, time, random, sys
from tqdm import tqdm
import astropy.stats
import astropy.io.fits as fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy import modeling
from astropy import constants as const
from astropy.stats import sigma_clipped_stats
from scipy import signal
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
from scipy.integrate import trapz, simps, quad
from sedpy import observate
import sedpy, emcee, corner
from multiprocessing import Pool
import warnings
warnings.simplefilter('ignore')
from astropy.cosmology import FlatLambdaCDM
cosmo2 = FlatLambdaCDM(H0=70, Om0=0.3)

def plot_all(objid, plt_jy=True, sdir=None, imhead=None, show=None, add_text=None):
    
    if objid not in fsed['objid']:
        print('no sed for obj {}!')
        return None
    
    _idx = np.squeeze(np.where(fsed['objid']==objid))

    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    gs.update(hspace=0)
    resid = fig.add_axes([0.1,0.1,0.4,0.25])
    phot = fig.add_axes([0.1,0.35,0.4,0.5])
    
    ##### SED
    zred = fsed['zred'][_idx]
    print('zml:', zred)
    mu = fsed['mu'][_idx]
    obsmags = fsed['obsmags'][_idx]
    obsunc = fsed['obsunc'][_idx]
    wavspec = fsed['wavspec'] * (1+zred)
    
    weff = fsed['weff'] / 1e4
    wavspec = wavspec / 1e4
    
    modmags = fsed['modmags'][_idx] * mu
    modspec = fsed['modspec'][_idx] * mu
    
    mask = np.isfinite(obsmags)
    obsmags = obsmags[mask]
    obsunc = obsunc[mask]
    weff = weff[mask]
    modmags = modmags[mask]
    obsunc = np.clip(obsunc, a_min=obsmags*0.05, a_max=None)
        
    if plt_jy:
        obsmags *= 3631
        obsunc *= 3631
        modmags *= 3631
        modspec *= 3631

    phot.errorbar(weff, modmags, fmt='o', color='firebrick', label='model photometry', zorder=100,
                  elinewidth=1, mec='k', mew=0.2)

    phot.errorbar(weff, obsmags, yerr=obsunc, color='black', fmt='o', label='observed photometry', zorder=101)
    phot.plot(wavspec, modspec, '-', color='firebrick', label = 'model spectrum', zorder=-100)
    
    xmin, xmax = weff.min()*0.8, weff[:-1].max()*1.5
    phot.set_xlim(xmin,xmax)
    ymin = np.nanmin(obsmags[obsmags>0])*0.3
    ymax = np.nanmax(obsmags)*2
    phot.set_ylim(ymin,ymax)
    resid.set_xlim(xmin,xmax)
    
    if (obsmags < 0).any():
        downarrow = [u'\u2193']
        y0 = 10**((np.log10(ymax) - np.log10(ymin))/20.)*ymin
        for x0 in weff[obsmags < 0]: phot.plot(x0, y0, linestyle='none',marker=u'$\u2193$',markersize=16,mew=0.5,mec='k',color='k')       
    #phot.legend(loc='lower right')
    
    photchi = (modmags-obsmags)/obsunc
    resid.plot(weff,photchi,'o',color='firebrick')
    resid.axhline(0.0,linestyle=':', color='grey')

    ymin, ymax = resid.get_ylim()
    yl = np.nanmax([np.abs(ymin),np.abs(ymax)])
    resid.set_ylim(-yl,yl)

    phot.text(0.97,0.92,str(objid),fontsize=18,transform=phot.transAxes,ha='right',va="top",weight='bold')
    phot.text(0.97,0.82,add_text,fontsize=18,transform=phot.transAxes,ha='right',va="top")
    lblsize = 16
    resid.set_xlabel(r'$\lambda_{\rm{observed}} \; [\mu m]$',fontsize=lblsize)
    resid.set_ylabel(r'$\chi$',fontsize=lblsize)
    ylab = 'Flux density [maggies]'
    if plt_jy:
        ylab = 'Flux density [Jy]'
    phot.set_ylabel(ylab,fontsize=lblsize)
    phot.set_yscale('log',nonpositive='clip')
    phot.set_xscale('log',nonpositive='clip')
    resid.set_xscale('log',nonpositive='clip',subs=(1,2,3,4,5,6,8))
    resid.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
    resid.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
    phot.set_xticklabels([])
    fig.tight_layout()
    
    ##### SFH
    idx_in_sfh = np.where(fsfh['objid']==objid)[0][0]
    sfh = fsfh['sfh'][idx_in_sfh]
    i50 = sfh.shape[0]//2 # idx of the median SFH
    
    tbins = sfh.shape[1]
    age = np.logspace(1, fsfh['agebins_max'][idx_in_sfh], tbins)/1e9
    
    axsfh = fig.add_axes([0.6,0.1,0.4,0.75])
    axsfh.step(age, sfh[i50, :], color='k')
    axsfh.fill_between(age, sfh[i50-1, :], sfh[i50+1, :], alpha=0.5, color='gray', step='pre')
    axsfh.fill_between(age, sfh[i50-2, :], sfh[i50+2, :], alpha=0.2, color='gray', step='pre')

    axsfh.set_xlim(1e-3, np.max(age))
    axsfh.set_xscale('log')
    # plt.yscale('log')
    fs = 18
    axsfh.set_xlabel(r'Lookback time [Gyr]', fontsize=fs)
    axsfh.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]', fontsize=fs)
    
    ##### redshift
    idx_in_z = np.where(fchain['objid']==objid)[0][0]
    thetaidx_in_z = dict(zip(fchain['theta_labels'], np.arange(len(fchain['theta_labels']))))
    
    _chain = fchain['chains'][idx_in_z]    
    _chain_for_z = _chain[:,[thetaidx_in_z['zred']]]
    axz = fig.add_axes([0.12,0.65,0.12,0.15])
    axz.set_yticks([])
    axz.hist(_chain_for_z,color="black",density=True)    
    
    del _chain, _chain_for_z
    
    ##### image
    if objid not in id_mega:
        print('no sed for obj {}!')
    crossid_here = np.where(id_mega==float(objid))[0][0]
    ra_here = ra_mega[crossid_here]
    dec_here = dec_mega[crossid_here]
    
    if sfh[i50, 0]<0.5*np.nanmax(sfh[i50+2, :])*1.1:
        thumb = fig.add_axes([0.62,0.5,0.12,0.3])
    else:
        thumb = fig.add_axes([0.62,0.15,0.12,0.3])
        
    y_pix, x_pix = wcs_jwst.wcs_world2pix(ra_here,dec_here,0)
    data_around = data[int(x_pix-10/pixscale):int(x_pix+10/pixscale),int(y_pix-10/pixscale):int(y_pix+10/pixscale)]
    data_show = data[int(x_pix-2/pixscale):int(x_pix+2/pixscale),int(y_pix-2/pixscale):int(y_pix+2/pixscale)]
    imsize = np.shape(data_show)[0]
    mean_pix, median_pix, stddev_pix = sigma_clipped_stats(data_around)

    if np.isnan(stddev_pix)==True:
        for k in range(10):
            std = np.nanstd(data_around)
            data[np.where(data>3*std)]=np.nan
        stddev_pix = std

    thumb.set_xticks([])
    thumb.set_yticks([])

    thumb.imshow(data_show,vmin=-2*stddev_pix,vmax=10*stddev_pix,cmap="viridis")
    clip = imsize/2-0.5
    color1 = 'red'
    thumb.plot([clip,clip],[clip*0.7,clip*0.2],color=color1,lw=0.5)
    thumb.plot([clip,clip],[clip*1.3,clip*1.8],color=color1,lw=0.5)
    thumb.plot([clip*0.7,clip*0.2],[clip,clip],color=color1,lw=0.5)
    thumb.plot([clip*1.3,clip*1.8],[clip,clip],color=color1,lw=0.5)
    
    del data_around, data_show
    
    if sdir is not None:
        plt.savefig(sdir+'full_{0}_{1}.png'.format(objid,imhead), bbox_inches='tight')
    if show is None:
        plt.ioff()
    #plt.show()

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

ffsed = cat_dir+'ancillaries/spec_phisfh.h5'.format(fver)
ffsfh = cat_dir+'ancillaries/sfhs_phisfh.h5'.format(fver)
ffchain = cat_dir+'ancillaries/chains_phisfh.h5'.format(fver)
fsed = np.load(ffsed, allow_pickle=True)
fsfh = np.load(ffsfh, allow_pickle=True)
fchain = np.load(ffchain, allow_pickle=True)

print(id_mega_photo)
