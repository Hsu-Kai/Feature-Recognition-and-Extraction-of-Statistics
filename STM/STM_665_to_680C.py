import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

img = plt.imread('STM_665_to_680C.png')[:,:,0]  

sig = 1.5
imf = gaussian_filter(img, sigma=sig, mode='mirror', order=0) # filter/smooth

if True:
    fig, ax = plt.subplots(1, 1)
    ax.imshow(imf, cmap='gray')
    plt.show()

gx = np.array([1, 0, -1]) * np.array([1, 2, 1])[:, None] 
gy = gx.T.copy()

Gx, Gy = [convolve(imf, g) for g in (gx, gy)]

G = np.sqrt(Gx**2 + Gy**2)
Th = np.arctan2(Gx, Gy)
Thm = np.mod(Th, np.pi/3)

fig, axes = plt.subplots(2, 2, figsize=[9, 7])
things = Gx, Gy, G, Thm
cmaps = 3*['gray'] + ['jet']
for ax, thing, cmap in zip(axes.flatten(), things, cmaps):
    ax.imshow(thing, cmap=cmap)
plt.show()

plt.imshow(G, cmap='gray')
plt.show()

sig = 1.5
Gf = gaussian_filter(G, sigma=sig, mode='mirror', order=0) # filter/smooth

if True:
    fig, ax = plt.subplots(1, 1)
    ax.imshow(G>0.25, cmap='gray')
    plt.show()

thresholds = np.arange(0.15, 0.3, 0.05)
answers = []
for G_threshold in thresholds:
    keep = G > G_threshold
    anglez = np.mod(np.degrees(Th[keep]), 360)
    a, b = np.histogram(anglez, bins=range(0, 362, 2))
    answers.append(a)

if True:
    for a in answers:
        plt.plot(b[:-1], a)
        plt.xticks([0, 30, 90, 150, 210, 270, 330])
    plt.show()


