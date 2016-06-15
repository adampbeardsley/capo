# A script to create Tsky vs LST

from astropy.io import fits
from pygsm import GSMObserver
import numpy as np
from scipy import interpolate
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import healpy as hp

hera_beam_file = '/home/beards/code/python/PRISim/prisim/data/beams/HERA_HFSS_X4Y2H_4900.hmap'

df = 1.5625
freqs = np.arange(100.0+df/2.0, 200.0, df)
hours = np.arange(0.0, 24.0, .5)
lsts = np.zeros_like(hours)
nside = 32
pols = ['X']  # Only have X beam right now
HERA_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))
# Set up the observer
(latitude, longitude, elevation) = ('-30.7224', '21.4278', 1100)
ov = GSMObserver()
ov.lon = longitude
ov.lat = latitude
ov.elev = elevation

# Read in HERA beam data
hera_beam = {}
for poli, pol in enumerate(pols):
    temp_im = fits.getdata(hera_beam_file, extname='BEAM_{0}'.format(pol))
    temp_f = fits.getdata(hera_beam_file, extname='FREQS_{0}'.format(pol))
    # Interpolate to the desired frequencies
    func = interpolate.interp1d(temp_f, temp_im, kind='cubic', axis=1)
    hera_beam[pol] = func(freqs)

fig = plt.figure("Tsky calc")
for poli, pol in enumerate(pols):
    for fi, freq in enumerate(freqs):
        print 'Forming Tsky for frequency ' + str(freq) + ' MHz.'
        # Rotate and project hera beam
        beam = hp.orthview(hera_beam[pol][:, fi], rot=[0, 90], fig=fig.number,
                           xsize=400, return_projected_map=True, half_sky=True)
        beam[np.isinf(beam)] = np.nan
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
            HERA_Tsky[poli, fi, ti] = np.nanmean(beam * sky) / np.nanmean(beam)
