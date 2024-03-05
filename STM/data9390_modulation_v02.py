import numpy as np
import matplotlib.pyplot as plt 
import rhksm4
import glob
from scipy.ndimage import gaussian_filter


class Scan():
    goodies = (('xscale', 'RHK_Xscale'), ('yscale','RHK_Yscale'), 
               ('zscale', 'RHK_Zscale'), ('bias', 'RHK_Bias'), ('period', 'RHK_Period'),
               ('current', 'RHK_Current'), ('angle', 'RHK_Angle'),
               ('over', 'RHK_OverSamplingCount'), ('xsize', 'RHK_Xsize'), ('ysize', 'RHK_Ysize'), 
               ('session_text', 'RHK_SessionText'), ('user_text', 'RHK_UserText'),
               ('date', 'RHK_Date'), ('time', 'RHK_Time'),
               ('statuschanneltext', 'RHK_StatusChannelText'))

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return ('{self.name}'.format(self=self))

def flatten_five(array):
    """returns 0 through 4th order flattenings of array,
       I am sure there is a more elegant way to do this"""
    H = array.copy()
    x, y = [np.linspace(-1, 1, s) for s in H.shape]
    X, Y = np.meshgrid(x, y, copy=False)

    X = X.flatten()
    Y = Y.flatten()
    C = np.ones_like(X)

    A0 = np.array([C]).T
    A1 = np.array([X, Y, C]).T
    A2 = np.array([X**2, X*Y, Y**2, X, Y, C]).T
    A3 = np.array([X**3, X**2*Y, X*Y**2, Y**3, X**2, X*Y, Y**2, X, Y, C]).T
    A4 = np.array([X**4, X**3*Y, X**2*Y**2, X*Y**3, Y**4,
                   X**3, X**2*Y, X*Y**2, Y**3, X**2, X*Y, Y**2, X, Y, C]).T
    B = H.flatten()

    coeff0, r, rank, ss = np.linalg.lstsq(A0, B, rcond=None)
    coeff1, r, rank, ss = np.linalg.lstsq(A1, B, rcond=None)
    coeff2, r, rank, ss = np.linalg.lstsq(A2, B, rcond=None)
    coeff3, r, rank, ss = np.linalg.lstsq(A3, B, rcond=None)
    coeff4, r, rank, ss = np.linalg.lstsq(A4, B, rcond=None)

    c1 = coeff0[0]
    F0 = c1 * np.ones_like(X)

    c1, c2, c3 = coeff1
    F1 = c1*X + c2*Y + c3

    c1, c2, c3, c4, c5, c6 = coeff2
    F2 = c1*X**2 + c2*X*Y + c3*Y**2 + c4*X + c5*Y + c6

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = coeff3
    F3 = (c1*X**3 + c2*X**2*Y + c3*X*Y**2 + c4*Y**3 +
          c5*X**2 + c6*X*Y + c7*Y**2 + c8*X + c9*Y + c10)

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = coeff4
    F4 = (c1*X**4 + c2*X**3*Y + c3*X**2*Y**2 + c4*X*Y**3 + c5*Y**4 +
          c6*X**3 + c7*X**2*Y + c8*X*Y**2 + c9*Y**3 +
          c10*X**2 + c11*X*Y + c12*Y**2 + c13*X + c14*Y + c15)

    Flat0 = F0.reshape(*H.shape)
    Flat1 = F1.reshape(*H.shape)
    Flat2 = F2.reshape(*H.shape)
    Flat3 = F3.reshape(*H.shape)
    Flat4 = F4.reshape(*H.shape)
    Flattened0 = H - Flat0
    Flattened1 = H - Flat1
    Flattened2 = H - Flat2
    Flattened3 = H - Flat3
    Flattened4 = H - Flat4
    
    return (Flattened0, Flattened1, Flattened2, Flattened3, Flattened4)

# fnames = glob.glob('*.SM4')

fnames = ['data9390.SM4']

for fname in fnames:
    name = fname.split('.SM4')[0]
    print('fname: ', fname)
    
    f = rhksm4.load(fname)

    scans = []

    for i in range(len(f._pages)):

        array, infodic = f[i].data, f[i].attrs

        if (isinstance(array, np.ndarray) and array.ndim==2 and
            isinstance(infodic, dict)):
            
            scan = Scan(name + '_' + str(100+i)[1:])
            scans.append(scan)
            scan.array = array
            scan.farray = array.astype(float)
            scan.ffarrays = flatten_five(scan.farray)
            scan.infodic = infodic
            
            for a, b in scan.goodies:
                if b in scan.infodic:
                    setattr(scan, a, infodic[b])
                else:
                    setattr(scan, a, None)
            scan.ffsarrays = [a * scan.zscale for a in scan.ffarrays] # new!
    N = len(scans)
    if N >= 1:
        fig, axes = plt.subplots(1, N, figsize=[12, 5])
        for ax, scan in zip(axes, scans):
            sy, sx = scan.array.shape
            extent = [0, scan.xscale * sx, scan.yscale * sy, 0]
            ax.imshow(scan.ffarrays[1], extent=extent, cmap='afmhot') 
    fig.suptitle(fname)
    plt.show()

extent_Angstroms = [0, 1E+10 * scan.xscale * sx, 1E+10 * scan.yscale * sy, 0]

s0, s1 = scans

# use only planar flattening to avoid removing any modulation

arrays = [s.ffsarrays[1] for s in scans]

# divide each line by its mean or median

arrays_means = [a - a.mean(axis=1)[:, None] for a in arrays]
arrays_medians = [a - np.median(a, axis=1)[:, None] for a in arrays]

fig, rows = plt.subplots(2, 2, figsize=[14, 7.5])
titles = (('forward, subtract row means', 'backwards, subtract row means'),
          ('forward, subtract row medians', 'backwards, subtract row medians'))

for row, pairs, t_pairs, in zip(rows, (arrays_means, arrays_medians), titles):
    for ax, array, title in zip(row, pairs, t_pairs):
        ax.imshow(array, cmap='afmhot', extent=extent_Angstroms)
        ax.set_title(title)
fig.suptitle(fname)
plt.show()

# Gaussian filter to reduce but not removes the TMD ~ 3A lattice periodicity

sigma_Angstroms = 1. # Angstroms

sigma_pixels = (sigma_Angstroms/1E+10) / abs(scan.xscale)  # meters / meters per pixel = pixels

arrays_means_gf = [gaussian_filter(img, sigma=sigma_pixels, mode='mirror', order=0)
                   for img in arrays_means]

arrays_medians_gf = [gaussian_filter(img, sigma=sigma_pixels, mode='mirror', order=0)
                   for img in arrays_medians]

fig, rows = plt.subplots(2, 2, figsize=[14, 7.5])
titles = (('forward, subtract row means then filter (Å)',
           'backwards, subtract row means then filter (Å)'),
          ('forward, subtract row medians then filter (Å)',
           'backwards, subtract row medians then filter (Å)'))

for row, pairs, t_pairs, in zip(rows, (arrays_means_gf, arrays_medians_gf), titles):
    for ax, array, title in zip(row, pairs, t_pairs):
        im = ax.imshow(1E+10*array, cmap='afmhot', extent=extent_Angstroms)
        clb = fig.colorbar(im, ax=ax)
        ax.set_title(title)
fig.suptitle(fname)
plt.show()

# They all look the same (in low frequency) so let's just choose one
# and see what the FT look like.

array = arrays_means_gf[0]

# let's re-flatten this frequency-filtered image
array_flattened = flatten_five(array)[2]


s0, s1 = array_flattened.shape
hwy, ewy, hwx, ewx = s0 >> 1, s0 >> 3, s1 >>1, s1 >> 3 # we'll use in the FT plots
window = np.hanning(s0)[:, None] * np.hamming(s1)# vhttps://en.wikipedia.org/wiki/Window_function

arrays = (array_flattened, array_flattened * window)

arrays = [a - a.mean() for a in arrays] # this lowers the uninteresting blob at zero frequency

titles = ('array', 'array_windowed')
fig, axes = plt.subplots(1, 2, figsize=[12, 7.5])
for array, title, ax in zip(arrays, titles, axes):
    ax.imshow(array, cmap='afmhot', extent=extent)
    ax.set_title(title)
plt.show()

# now lets calculate the power spectrum by taking the Fourier transform (complex)
# then squaring the absolute value for power

fts = [np.fft.fftshift(np.fft.fft2(array)) for array in arrays]
powers = [np.abs(ft)**2 for ft in fts]
powers = [p/p.max() for p in powers]  # normalize so we can use vmin/vmax on log10(power)
log_powers = [np.log10(p) for p in powers]

fig, axes = plt.subplots(2, 2, figsize=[12, 7.5])
cmaps = 'afmhot', 'viridis'
vminmaxes = ((None, None), (-5, 0))
zoomed_log_powers = [p[hwy-ewy:hwy+ewy+1, hwx-ewx:hwx+ewx+1] for p in log_powers]
things = [arrays, zoomed_log_powers]

for two, axez, cmap, vminmax in zip(things, axes, cmaps, vminmaxes):
    vmin, vmax = vminmax
    for thing, ax in zip(two, axez):
        ax.imshow(thing, cmap=cmap, vmin=vmin, vmax=vmax)
plt.show()
    
