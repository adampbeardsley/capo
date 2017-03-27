# A script to create Tsky vs LST

from astropy.io import fits
from pygsm import GSMObserver
import numpy as np
from scipy import interpolate
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import healpy as hp

calc_paper = False

hera_beam_file = '/data4/beards/instr_data/HERA_HFSS_X4Y2H_4900.hmap'

df = 1.5625  # 100 MHz / 64 averaged channels
freqs = np.arange(100.0 + df / 2.0, 200.0, df)
hours = np.arange(0.0, 24.0, .5)
lsts = np.zeros_like(hours)
pols = ['X', 'Y']  # Only have X beam, but try rotating 90 degrees for Y
HERA_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))

# Read in HERA beam data, just use full sky for paper
hera_beam = {}
# Only have X right now, will rotate later
hera_im = fits.getdata(hera_beam_file, extname='BEAM_{0}'.format('X'))
nside = hp.npix2nside(hera_im.shape[0])
temp_f = fits.getdata(hera_beam_file, extname='FREQS_{0}'.format('X'))
# Interpolate to the desired frequencies
func = interpolate.interp1d(temp_f, hera_im, kind='cubic', axis=1)
for pol in pols:
    hera_beam[pol] = func(freqs)

if calc_paper:
    PAPER_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))

    # Read in paper beam data
    paper_beam = {}
    paper_im = np.zeros_like(hera_im)
    for fi, f in enumerate(temp_f):
        beam_file = '/data4/beards/instr_data/paper_beam_{0}.fits'.format(np.int(f))
        paper_im[:, fi] = hp.ud_grade(hp.read_map(beam_file), nside)
    # Interpolate to the desired frequencies
    func = interpolate.interp1d(temp_f, paper_im, kind='cubic', axis=1)
    for pol in pols:
        paper_beam[pol] = np.abs(func(freqs))**2.0


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
        if calc_paper:
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
            if calc_paper:
                PAPER_Tsky[poli, fi, ti] = np.nanmean(pbeam * sky) / np.nanmean(pbeam)

inds = np.argsort(lsts)
lsts = lsts[inds]
HERA_Tsky = HERA_Tsky[:, :, inds]

Tsky_file = '/data4/beards/HERA_IDR1_analysis/HERA_Tsky.npz'
np.savez(Tsky_file, HERA_Tsky=HERA_Tsky, freqs=freqs, lsts=lsts)
if calc_paper:
    PAPER_Tsky = PAPER_Tsky[:, :, inds]
    Tsky_file = '/data4/beards/HERA_IDR1_analysis/PAPER_Tsky.npz'
    np.savez(Tsky_file, PAPER_Tsky=PAPER_Tsky, freqs=freqs, lsts=lsts)
