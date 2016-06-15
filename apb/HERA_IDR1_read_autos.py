import numpy as np
from uvdata.uv import UVData
import capo
import aipy
from glob import glob

nant = 128
chanave = 16

xxglob = '/data6/HERA/data/2457458/*xx.uv'
yyglob = '/data6/HERA/data/2457458/*yy.uv'

xxfilenames = glob(xxglob)
yyfilenames = glob(yyglob)

xxt, xxd, xxf = capo.arp.get_dict_of_uv_data(xxfilenames, polstr='xx', antstr='auto')
nt, nchan = xxd[0, 0]['xx'].shape
# Do some coarse averaging
xxd_ave = np.zeros((nant, nt, nchan / chanave), dtype=np.float64)
for ant in xrange(128):
    for chan in xrange(nchan / chanave):
        xxd_ave[ant, :, chan] = np.mean(xxd[ant, ant]['xx'][:, (chan * chanave):((chan + 1) * chanave)], axis=1)
