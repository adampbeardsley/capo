# A script to create Tsky vs LST

from astropy.io import fits
from pygsm import GSMObserver
import numpy as np
from scipy import interpolate
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import healpy as hp

hera_beam_file = '/data2/beards/instr_data/HERA_HFSS_X4Y2H_4900.hmap'

df = 1.5625  # 100 MHz / 64 averaged channels
freqs = np.arange(100.0 + df / 2.0, 200.0, df)
hours = np.arange(0.0, 24.0, .5)
lsts = np.zeros_like(hours)
nside = 32
pols = ['X', 'Y']  # Only have X beam, but try rotating 90 degrees for Y
HERA_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))
PAPER_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))
paper_width0 = np.pi / 2.0  # FWHM in radians

# Read in HERA beam data, just use full sky for paper
hera_beam = {}
paper_beam = {}
for poli, pol in enumerate(pols):
    # Only have X right now, will rotate later
    temp_im = fits.getdata(hera_beam_file, extname='BEAM_{0}'.format('X'))
    temp_f = fits.getdata(hera_beam_file, extname='FREQS_{0}'.format('X'))
    # Interpolate to the desired frequencies
    func = interpolate.interp1d(temp_f, temp_im, kind='cubic', axis=1)
    hera_beam[pol] = func(freqs)
    paper_beam[pol] = np.zeros_like(hera_beam[pol])
    paper_beam[pol][0, :] = 1
    for fi, freq in enumerate(freqs):
        paper_beam[pol][:, fi] = hp.smoothing(paper_beam[pol][:, fi],
                                              fwhm=paper_width0 * freqs.mean() / freq,
                                              verbose=False)

# Set up the observer
(latitude, longitude, elevation) = ('-30.7224', '21.4278', 1100)
ov = GSMObserver()
ov.lon = longitude
ov.lat = latitude
ov.elev = elevation

fig = plt.figure("Tsky calc")
for poli, pol in enumerate(pols):
    for fi, freq in enumerate(freqs):
        print 'Forming HERA Tsky for frequency ' + str(freq) + ' MHz.'
        # Rotate and project hera beam (Need to figure out how to do this w/o making figure)
        pol_ang = 90 * (1 - poli)  # Extra rotation for X
        hbeam = hp.orthview(hera_beam[pol][:, fi], rot=[pol_ang, 90], fig=fig.number,
                            xsize=400, return_projected_map=True, half_sky=True)
        hbeam[np.isinf(hbeam)] = np.nan
        pbeam = hp.orthview(paper_beam[pol][:, fi], rot=[pol_ang, 90], fig=fig.number,
                            xsize=400, return_projected_map=True, half_sky=True)
        pbeam[np.isinf(pbeam)] = np.nan
        for ti, t in enumerate(hours):
            plt.clf()
            dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                          np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
            lsts[ti] = Time(dt).sidereal_time('apparent', longitude).hour
            ov.date = dt
            ov.generate(freq)
            d = ov.view(fig=fig.number)
            sky = hp.orthview(d, fig=fig.number, xsize=400, return_projected_map=True,
                              half_sky=True)
            sky[np.isinf(sky)] = np.nan
            HERA_Tsky[poli, fi, ti] = np.nanmean(hbeam * sky) / np.nanmean(hbeam)
            PAPER_Tsky[poli, fi, ti] = np.nanmean(pbeam * sky) / np.nanmean(pbeam)

inds = np.argsort(lsts)
lsts = lsts[inds]
HERA_Tsky = HERA_Tsky[:, :, inds]
PAPER_Tsky = PAPER_Tsky[:, :, inds]

Tsky_file = '/data2/beards/tmp/HERA_Tsky.npz'
np.savez(Tsky_file, HERA_Tsky=HERA_Tsky, freqs=freqs, lsts=lsts)
Tsky_file = '/data2/beards/tmp/PAPER_Tsky.npz'
np.savez(Tsky_file, PAPER_Tsky=PAPER_Tsky, freqs=freqs, lsts=lsts)
