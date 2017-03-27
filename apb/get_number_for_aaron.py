# A script to get absolute gain factor for the HERA dishes from IDR1.
# Must run HERA_IDR1_read_autos.py and Tsky_v_LST.py first.

import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt
import aipy


def curve_to_fit(lsts, gain, gain_slope, rxr_amp):
    # Define curve to fit data to
    # gain, gain_slope, rxr_amp, and rxr_slope are the parameters to fit
    global interp_values
    lsts_shifted = lsts - lsts.min()
    return (gain * interp_values + gain_slope * lsts_shifted * interp_values + rxr_amp)


def match_model_to_data(lsts, pol, fi, params):
    # Take model (params correspond to fit parameters from curve_to_fit)
    # and create curve that should match the observed data (ie. apply gains to theory)
    global interp_func
    interp_values = interp_func(lsts)[pol, fi, :]
    lsts_shifted = lsts - lsts.min()
    return (params[0] * interp_values + params[1] * lsts_shifted * interp_values +
            params[2])


def match_data_to_model(lsts, data, params):
    # Use model (params correspond to fit parameters from curve_to_fit)
    # to correct the data to match the theoretical Tsky (ie. apply gains to data)
    lsts_shifted = lsts - lsts.min()
    return ((data - params[2]) / (params[0] + params[1] * lsts_shifted))

# Only run this script for HERA antennas
HERA_list = [9, 10, 20, 22, 31, 43, 53, 64, 65, 72, 80, 81, 88, 89, 96, 97, 104, 105, 112]
nant = len(HERA_list)
pols = ['X', 'Y']
npol = len(pols)
save_plots = True

# Load sky model from previous script:
# I ran the Tsky calc twice - once with the beam as is in the file, once by squaring
# the beam. I'm fairly certain now (by looking at PRISim code) that the file I
# have is already in power units, so the first version should be used.
Tsky_file = '/data4/beards/HERA_IDR1_analysis/HERA_Tsky.npz'
# Tsky_file = '/data4/beards/HERA_IDR1_analysis/HERA_Tsky_updated.npz'
data = np.load(Tsky_file)
freqs = data['freqs']  # These are already made to fit the data
model_lsts = data['lsts']
model_HERA_Tsky = data['HERA_Tsky']

# Load in data from previous script
autos_file = '/data4/beards/HERA_IDR1_analysis/IDR1_autos.npz'
data = np.load(autos_file)
data_lsts = data['lsts']
data_ave = data['data_ave']

# Fit auto data to model
auto_fits = np.zeros((npol, nant, len(freqs), 3))  # gain_amp, gain_slope, rxr_amp
covs = np.zeros((npol, nant, len(freqs), 3, 3))
interp_func = interpolate.interp1d(model_lsts, model_HERA_Tsky, kind='cubic', axis=2)
for pol in xrange(npol):
    interp_array = interp_func(data_lsts[pol])
    for fi, freq in enumerate(freqs):
        interp_values = interp_array[pol, fi, :]
        for anti, ant in enumerate(HERA_list):
            out = curve_fit(curve_to_fit, data_lsts[pol], data_ave[pol][ant, :, fi])
            auto_fits[pol, anti, fi, :] = out[0]
            covs[pol, anti, fi, :, :] = out[1]
            # auto_fits[pol, anti, fi, :] = curve_fit(curve_to_fit, data_lsts[pol],
                                                    # data_ave[pol][ant, :, fi])[0]

# Get gains and receiver temperatures
gains = np.zeros((npol, nant, len(freqs), len(data_lsts[0])))
rxr_temp = np.zeros((npol, nant, len(freqs), len(data_lsts[0])))
lsts_shifted = [data_lsts[0] - data_lsts[0].min(), data_lsts[1] - data_lsts[1].min()]
for pol in xrange(npol):
    for ant in xrange(nant):
        gains[pol, ant, :, :] = (auto_fits[pol, ant, :, 0].reshape(-1, 1) +
                                 auto_fits[pol, ant, :, 1].reshape(-1, 1) *
                                 lsts_shifted[pol].reshape(1, -1))
        rxr_temp[pol, ant, :, :] = auto_fits[pol, ant, :, 2].reshape(-1, 1) / gains[pol, ant, :, :]

# rxr_temp is in K already. Need to convert gains to give Jy
# Jy = 2761.3006 * (T/K) * (m^2/Ae)
beam_file = '/data4/beards/instr_data/HERA_HFSS_X4Y2H_4900.hmap'
beam = fits.getdata(hera_beam_file, extname='BEAM_{0}'.format('X'))
beam_f = fits.getdata(hera_beam_file, extname='FREQS_{0}'.format('X'))
func = interpolate.interp1d(beam_f, beam, kind='cubic', axis=1)
npix = beam.shape[0]
Ae = (aipy.const.c / (100.0 * freqs * 1.0e6))**2.0 / (4 * np.pi / npix * np.sum(func(freqs), axis=0))
K_to_Jy = 2761.3006 / Ae
gains_Jy = np.sqrt(gains * K_to_Jy.reshape(1, 1, -1, 1))

if save_plots:
    # first plot the actual fits
    outdir = '/data4/beards/HERA_IDR1_analysis/plots/'
    finds = [10, 32, 54]  # beginning, middle, end, but not at the very edges
    dark_colors = ['blue', 'green', 'red']
    light_colors = ['dodgerblue', 'lightgreen', 'salmon']
    fig = plt.figure('auto fits')
    for pol in xrange(npol):
        for anti, ant in enumerate(HERA_list):
            # Fits plot
            fig.clf()
            tittext = 'Antenna ' + str(ant) + pols[pol] + ' (HERA)'
            for i, fi in enumerate(finds):
                label = str(freqs[fi]) + ' MHz'
                plt.plot(data_lsts[pol], match_data_to_model(data_lsts[pol], data_ave[pol][ant, :, fi],
                                                             auto_fits[pol, anti, fi, :]), '.',
                         ms=5, color=dark_colors[i], label=label)
                plt.plot(data_lsts[pol], interp_func(data_lsts[pol])[pol, fi, :],
                         color=light_colors[i])
            plt.ylim([0, 1.3 * np.max(interp_func(data_lsts[pol])[pol, finds[0], :])])
            plt.xlabel('LST (Hours)')
            plt.ylabel('Tsky')
            plt.title(tittext)
            plt.legend()
            outfile = outdir + 'Tsky_v_LST_fit' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile)

            # Apply gains to model
            fig.clf()
            for i, fi in enumerate(finds):
                label = str(freqs[fi]) + 'MHz'
                plt.plot(data_lsts[pol], data_ave[pol][ant, :, fi], '.',
                         ms=5, color=dark_colors[i], label=label)
                plt.plot(data_lsts[pol], match_model_to_data(data_lsts[pol], pol,
                         fi, auto_fits[pol, anti, fi, :]), color=light_colors[i])
            plt.ylim([0, 1.3 * np.max(data_ave)])
            plt.xlabel('LST (Hours)')
            plt.ylabel('Data and gains applied to model')
            plt.title(tittext)
            plt.legend()
            outfile = outdir + 'data_v_LST_fit' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile)

            # Gain plot
            fig.clf()
            plt.imshow(gains_Jy[pol, anti, :, :].transpose(), aspect='auto', origin='lower',
                       extent=(freqs.min(), freqs.max(), data_lsts[pol].min(), data_lsts[pol].max()),
                       interpolation='none')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('LST (Hour)')
            plt.title(tittext)
            plt.colorbar(label='Gain (Jy units)')
            outfile = outdir + 'gain_waterfall' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile)

            fig.clf()
            plt.imshow(np.log10(rxr_temp[pol, anti, :, :]).transpose(), aspect='auto', origin='lower',
                       extent=(freqs.min(), freqs.max(), lsts[pol].min(), lsts[pol].max()),
                       interpolation='none')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('LST (Hour)')
            plt.title(tittext)
            plt.clim([0, 3 * np.log10(np.median(rxr_temp[pol, anti, :, :]))])
            plt.colorbar(label='Rxr Temp, Log10(K)')
            outfile = outdir + 'rxr_temp_waterfall' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile)

    # Final plot is gain vs Antenna and freq
    fig = plt.figure(figsize=(8, 5))
    mean_gains = np.sqrt(np.mean(gains, axis=3) * K_to_Jy.reshape(1, 1, -1))
    for pol in xrange(npol):
        fig.clf()
        for anti, ant in enumerate(HERA_list):
            plt.plot(freqs, mean_gains[pol, anti, :], label=str(ant))
        plt.xlabel('Freq (MHz)')
        plt.ylabel('Gain (Jy units)')
        ax = fig.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        outfile = outdir + 'avg_gains' + pols[pol] + '.png'
        plt.savefig(outfile)
