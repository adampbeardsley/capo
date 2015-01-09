''' This is a calibration file for data collected at PAPER in Karoo, South Africa
on JD 2455819 '''

import aipy as a, numpy as n,glob,ephem
import bm_prms as bm
import logging

prms = {
    'loc': ('-30:43:17.5', '21:25:41.9'), # KAT, SA (GPS)
    'antpos':{
        0:[147.659407413, 336.269469733, 264.566180759],
        1:[-120.566931266, -270.142735412, -201.208899961],
        2:[175.483874,-282.046474,309.593714],
        3:[-24.5939776529, -369.022493234, -35.9049669793],
        #--------------------------------------------------------
        4:[-135.977107,-65.373043,-223.715356],
        5:[-184.222167454,  60.9256169154, -307.675312464],
        6:[84.568610,-397.906007,151.703088],
        7:[60.9037241018, 422.222408268, 116.124879563],
        #--------------------------------------------------------
        8:[148.405177,-231.475974,263.305593],
        9:[-121.15655,-329.37685,-197.06224],
        10:[-28.800063,-420.849441,-43.564604],
        11:[-180.112865,-190.297251,-301.062917],
        #--------------------------------------------------------
        12:[161.032208592, 207.530151484, 286.703007713],
        13:[-79.830818,266.416356,-122.828467],
        14:[90.491568,406.666552,171.303074],
        15:[136.833937217,-349.10409, 256.16691],
        #========================================================
	    16:[75.008275,-366.663944,135.807286],
        17:[-170.082246,113.392564,-280.090332],
        18:[-173.308984588, -52.5844630491, -289.495946419],
        19:[35.6156894023, -76.4247822222, 68.8003235664],
        #-------------------------------------------------------
        20:[ 223.405495506, -111.371927382, 391.352958368],
        21:[ 211.984088554, -181.820834933, 372.672243377],
        22:[-52.9274701935, -409.284993158, -84.1268196632],
        23:[-75.327786,379.129646,-113.829018],
        #--------------------------------------------------------
        24:[-90.374808,3.548977,-144.207995],
        25:[-23.653561,-153.921245,-31.289596],
        26:[208.418197,189.287085,370.725255],
        27:[-22.2282015089, 311.926612877, -26.8228657991],
        #--------------------------------------------------------
        28:[-18.1453146192, 166.083642242, -21.2052534495],
        29:[89.6597220746, -22.1294190136, 162.698139384],
        30:[-139.053365,312.917932,-223.870462],
        31:[229.945829,48.161862,406.414507],
        #--------------------------------------------------------
        32:[-112.893563,109.228967,-182.880941],
        33:[121.355347,-319.429590,209.575748],
        34:[-1.186004,298.790781,-1.572735],
        35:[-150.754218,-224.460782,-258.594058],
        #--------------------------------------------------------
        36:[-148.166345,285.390149,-254.152706],
        37:[73.704070,-378.161280,127.753480],
        38:[183.238623,145.046381,314.997386],
        39:[201.110057,270.608943,345.388038],
        #--------------------------------------------------------
        40:[-187.753175,101.634584,-322.330703],
        41:[32.859445,-311.361270,57.492402],
        42:[111.791791,-360.752264,193.124569],
        43:[185.296482,12.473870,318.948404],
        #--------------------------------------------------------
        44:[66.840886,269.989165,115.139909],
        45:[208.327549,-181.024029,358.713760],
        46:[222.401981,114.559981,382.329808],
        47:[82.998742,-157.005822,143.375763],
        #-------------------------------------------------------
        48:[-123.364050,7.568406,-211.391982],
        49:[42.324815,-394.596554,73.800150],
        50:[155.428104,103.981800,267.545140],
        51:[4.002712,454.858259,7.086482],
        #-------------------------------------------------------
        52:[40.840441,382.998141,70.689703],
        53:[228.948582,78.038958,393.662509],
        54:[208.232148,171.396294,357.761446],
        55:[22.162702,221.120016,38.449461],
        #--------------------------------------------------------
        56:[-85.962903,360.456826,-147.018238],
        57:[-22.182170,447.517664,-37.585541],
        58:[-40.132905,-349.207661,-68.174661],
        59:[-38.864384,362.866457,-66.270033],
        #--------------------------------------------------------
        60:[134.062901,401.074665,230.468279],
        61:[-81.496611,-277.174777,-139.301327],
        62:[-161.608043,226.512058,-277.243397],
        63:[155.0510,-323.9710,288.9414],
    }, 
    'delays': {
     0 : {'x': -0.556, 'y':  3.554},
     1 : {'x': -0.474, 'y':  0.001},
     2 : {'x': -6.993, 'y': -6.830},
     3 : {'x': -0.741, 'y': -0.207},
     4 : {'x': -9.821, 'y':-12.876},
     5 : {'x':  5.374, 'y':  4.804},
     6 : {'x':-10.835, 'y':-14.388},
     7 : {'x':  3.101, 'y':  3.790},
     8 : {'x': -8.603, 'y': -8.814},
     9 : {'x': -0.061, 'y':  3.357},
    10 : {'x':-13.863, 'y':-14.429},
    11 : {'x':-15.116, 'y':-15.593},
    12 : {'x': -5.685, 'y': -6.590},
    13 : {'x':-13.523, 'y':-13.095},
    14 : {'x': -9.122, 'y':-13.898}, 
    15 : {'x':  2.761, 'y':  3.919},
    16 : {'x': 14.587, 'y': 15.185},
    17 : {'x': -0.398, 'y':  3.943},
    20 : {'x':  0.056, 'y':  0.423},
    21 : {'x': -0.363, 'y': -0.041},
    22 : {'x':  3.257, 'y':  1.353},
    23 : {'x':-12.735, 'y':-11.782},
    25 : {'x': -9.442, 'y':-11.594},
    26 : {'x':-10.692, 'y':-13.392},
    27 : {'x':  2.981, 'y': -1.705},
    28 : {'x':  3.034, 'y':  2.202},
    29 : {'x':  6.107, 'y':  2.315},
    30 : {'x': -7.793, 'y': -8.253},
    31 : {'x':-14.050, 'y':-12.730},
    #63 : {'x':-14.943, 'y':  9.606},
    },
    'amps': {
     0 : {'x': 16.518, 'y': 17.016},
     1 : {'x': 19.835, 'y': 18.889},
     2 : {'x': 15.451, 'y': 16.707},
     3 : {'x': 20.527, 'y': 19.395},
     4 : {'x': 20.891, 'y': 20.632},
     5 : {'x': 20.419, 'y': 17.828},
     6 : {'x': 16.816, 'y': 15.486},
     7 : {'x': 14.993, 'y': 11.885},
     8 : {'x': 16.660, 'y': 17.258},
     9 : {'x': 18.331, 'y': 17.405},
    10 : {'x': 18.703, 'y': 17.201},
    11 : {'x': 16.567, 'y': 16.503},
    12 : {'x': 20.657, 'y': 20.397}, 
    13 : {'x': 20.062, 'y': 17.895}, 
    14 : {'x': 17.471, 'y': 15.888},
    15 : {'x': 20.457, 'y': 20.898}, 
    16 : {'x': 18.612, 'y': 18.591},
    17 : {'x': 20.454, 'y': 20.060}, 
    20 : {'x': 19.135, 'y': 19.153},
    21 : {'x': 19.470, 'y': 17.392},
    22 : {'x': 17.140, 'y': 17.017},
    23 : {'x': 18.893, 'y': 13.733}, 
    25 : {'x': 16.133, 'y': 16.408},
    26 : {'x': 15.482, 'y': 16.745}, 
    27 : {'x': 13.521, 'y': 15.264}, 
    28 : {'x': 21.927, 'y': 20.900}, 
    29 : {'x': 22.011, 'y': 22.434},
    30 : {'x': 14.884, 'y': 16.354}, 
    31 : {'x': 20.767, 'y': 19.646},
    #63 : {'x': 14.850, 'y': 16.580},
    },
    'off':{
        },
    'bp_r':  n.array([-167333390752.98276, 198581623581.65594, -102487141227.4993, 30027423590.548084, -5459067124.669095, 630132740.98792362, -45056600.848056234, 1822654.0034047314, -31892.9279846797]) * 1.0178**0.5,
}

def get_aa(freqs):
    '''Return the AntennaArray to be used for simulation.'''
    location = prms['loc']
    antennas = []
    nants = len(prms['antpos'])
    for i in prms['antpos'].keys():
        beam = bm.prms['beam'](freqs,nside=32,lmax=20,mmax=20,deg=7)
        
        try: beam.set_params(bm.prms['bm_prms'])
        except(AttributeError): pass
        
        pos = prms['antpos'][i]
        
        try: dly = prms['delays'][i]
        except(KeyError): dly = {'x':0.,'y':0.}
        
        try: off = prms['off'][i]
        except(KeyError): off = {'x':0.,'y':0.}
        
        bp_r = {'x':prms['bp_r'],'y':prms['bp_r']}
        bp_i = {'x':[0.],'y':[0.]}
        
        try: amp=prms['amps'][i]
        except(KeyError): amp = {'x':20.,'y':20.}
       
        phsoff = {'x':[dly['x'],off['x']],'y':[dly['y'],off['y']]}
        
        antennas.append(
                a.pol.Antenna(pos[0],pos[1],pos[2], beam, phsoff=phsoff, amp=amp, bp_r = bp_r,bp_i=bp_i, lat=prms['loc'][0])
                )
    aa = a.pol.AntennaArray(prms['loc'], antennas)
    return aa

src_prms = {
    'pic' :{'jys': 357.,'ra':'5:19:49.70','dec':'-45:46.45.0','mfreq':0.150,'index':-2.19},
         #NVSS positions 
    'forA':{'jys': 301.,'ra':0.8807,'dec':-0.6480,'mfreq':0.150,'index':-1.273},
    'forB':{'jys': 182.,'ra':0.8904,'dec':-0.6504,'mfreq':0.150,'index':-1.066}, 
    'J0445-23':{'jys':50.,'ra':'4:45:43','dec':'-28:05:56','mfreq':0.150,'index':-1.},
    #'J0522-36':{'jys':96.,'ra':'5:23:43','dec':'-36:22:48','mfreq':0.150,'index':-1.},
    #'J0625-53':{'jys':60.,'ra':'6:25:11','dec':'-53:39:31','mfreq':0.150,'index':-1.},
}

def get_catalog(srcs=None, cutoff=None, catalogs=['helm','misc']):
    '''Return a catalog containing the listed sources.'''
    custom_srcs = ['J1347-603','J1615-610', 'J1336-340', 'J1248-412', 'J1531-423', 'J1359-415']
    if srcs is None:
        cat = a.src.get_catalog(srcs=srcs, cutoff=cutoff, catalogs=catalogs)
    else:
        cat = a.src.get_catalog(srcs=[s for s in srcs if not s in custom_srcs],
            cutoff=cutoff, catalogs=catalogs)
        for src in [s for s in srcs if s in custom_srcs]:
            cat[src] = a.fit.RadioFixedBody(0., 0., janskies=0., mfreq=.15, name=src)
    cat.set_params(src_prms)
    return cat

if __name__=='__main__':
    import sys, numpy as n
    if len(sys.argv)>1:
        print "loading catalog: ",sys.argv[1]
        cat = get_catalog(catalogs=[sys.argv[1]])
        names = [cat[src].src_name for src in cat]
        print "loaded",len(names)," sources"
        flx = [cat[src]._jys for src in cat]
        print names
        print "brightest source in catalog"
        print names[flx.index(n.max(flx))],n.max(flx)
        try: assert([cat[src].e_S_nu for src in cat])
        except(AttributeError): print "this catalog does not have flux errors"
