import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import copy

with open('sn_kde.pkl', 'rb') as f:
    sn_kde = pickle.load(f)

with open('red_kde.pkl', 'rb') as f:
    red_kde = pickle.load(f)

def get_rcf(ra, dec): # In degrees

    try:
        len(ra)
    except TypeError:
        ra = [ra]
        dec = [dec]

    l = np.radians(copy.deepcopy(ra))
    b = np.radians(copy.deepcopy(dec))

    for r in range(len(ra)):
        if l[r] > np.pi:
            l[r] -= 2*np.pi

    xy_sample = np.vstack((b, l)).T

    sn_zz = np.exp(sn_kde.score_samples(xy_sample))
    reds_zz = np.exp(red_kde.score_samples(xy_sample))

    for r in range(len(ra)):
        if dec[r] < -30:
            print('WARNING: RCF for (' + str(ra[r]) + ', ' + str(dec[r]) + ') might be invalid due to low declination.')
            continue

        if sn_zz[r] < 0.025:
            print('WARNING: RCF for (' + str(ra[r]) + ', ' + str(dec[r]) + ') might be invalid due to Milky Way foreground producing low SNe numbers in this region.')

    sn_zz_adj = sn_zz*4960
    reds_zz_adj = reds_zz*2237

    if len(sn_zz_adj) != 1:
        return reds_zz_adj/sn_zz_adj
    else:
        return float(reds_zz_adj/sn_zz_adj)
