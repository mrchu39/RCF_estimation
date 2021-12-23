import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import pickle
import copy
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308, Tcmb0=3)

with open('sn_kde.pkl', 'rb') as f:
    sn_kde = pickle.load(f)

with open('red_kde.pkl', 'rb') as f:
    red_kde = pickle.load(f)

with open('sn_kde_100.pkl', 'rb') as f:
    sn_kde_100 = pickle.load(f)

with open('red_kde_100.pkl', 'rb') as f:
    red_kde_100 = pickle.load(f)

with open('sn_kde_200-300.pkl', 'rb') as f:
    sn_kde_100_200 = pickle.load(f)

with open('red_kde_200-300.pkl', 'rb') as f:
    red_kde_100_200 = pickle.load(f)

with open('sn_kde_200-300.pkl', 'rb') as f:
    sn_kde_200_300 = pickle.load(f)

with open('red_kde_200-300.pkl', 'rb') as f:
    red_kde_200_300 = pickle.load(f)

with open('sn_kde_300-400.pkl', 'rb') as f:
    sn_kde_300_400 = pickle.load(f)

with open('red_kde_300-400.pkl', 'rb') as f:
    red_kde_300_400 = pickle.load(f)

with open('sn_kde_400.pkl', 'rb') as f:
    sn_kde_400 = pickle.load(f)

with open('red_kde_400.pkl', 'rb') as f:
    red_kde_400 = pickle.load(f)

def get_rcf_ind(ra, dec, redshift=None): # In degrees

    l = np.radians(copy.deepcopy(ra))
    b = np.radians(copy.deepcopy(dec))

    if l > np.pi:
        l -= 2*np.pi

    xy_sample = [[b, l]]

    sn_zz = np.exp(sn_kde.score_samples(xy_sample))
    reds_zz = np.exp(red_kde.score_samples(xy_sample))

    if dec < -30:
        print('WARNING: RCF for (' + str(ra) + ', ' + str(dec) + ') might be invalid due to low declination.')

    if sn_zz < 0.025:
        print('WARNING: RCF for (' + str(ra) + ', ' + str(dec) + ') might be invalid due to Milky Way foreground producing low SNe numbers in this region.')

    rcf = float(reds_zz*2237/(sn_zz*4960))

    if redshift != None and redshift != 'None':
        try:
            dist = cosmo.comoving_distance(float(redshift)).value

            if dist < 100:
                sn_zz_bin = np.exp(sn_kde_100.score_samples(xy_sample))
                reds_zz_bin = np.exp(red_kde_100.score_samples(xy_sample))
                size = 494
                red_size = 391
            elif dist >= 100 and dist < 200:
                sn_zz_bin = np.exp(sn_kde_100_200.score_samples(xy_sample))
                reds_zz_bin = np.exp(red_kde_100_200.score_samples(xy_sample))
                size = 1459
                red_size = 891
            elif dist >= 200 and dist < 300:
                sn_zz_bin = np.exp(sn_kde_200_300.score_samples(xy_sample))
                reds_zz_bin = np.exp(red_kde_200_300.score_samples(xy_sample))
                size = 1301
                red_size = 535
            elif dist >= 300 and dist < 400:
                sn_zz_bin = np.exp(sn_kde_300_400.score_samples(xy_sample))
                reds_zz_bin = np.exp(red_kde_300_400.score_samples(xy_sample))
                size = 1278
                red_size = 353
            elif dist >= 400:
                sn_zz_bin = np.exp(sn_kde_400.score_samples(xy_sample))
                reds_zz_bin = np.exp(red_kde_400.score_samples(xy_sample))
                size = 428
                red_size = 67

            binned_rcf = float(reds_zz_bin*red_size/(sn_zz_bin*size))

        except ValueError:
            binned_rcf = None

    else:
        binned_rcf = None

    return rcf, binned_rcf

def get_rcf(ra, dec, redshifts):

    try:
        len(ra)
    except ValueError:
        ra = [ra]
        dec = [dec]
        redshifts = [redshifts]

    rcfs = []
    binned_rcfs = []

    for r in range(len(ra)):
        rcf, binned_rcf = get_rcf_ind(ra[r], dec[r], redshifts[r])

        rcfs.append(rcf)
        binned_rcfs.append(binned_rcf)

    return pd.DataFrame(np.vstack((ra, dec, redshifts, rcfs, binned_rcfs)).T, columns=['RA', 'Dec', 'z', 'RCF', 'Binned RCF'])
