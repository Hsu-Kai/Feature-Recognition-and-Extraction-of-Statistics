import numpy as np
import matplotlib.pyplot as plt 
import rhksm4
import glob
from scipy.ndimage import gaussian_filter, convolve
import cv2

#fnames = glob.glob('*.SM4')

fnames = ['data7917.SM4']

for fname in fnames:
    name = fname.split('.SM4')[0]
    print('fname: ', fname)
    
    f = rhksm4.load(fname)

    N = len(f._pages)
    if N >= 1:
        fig, axes = plt.subplots(1, N, figsize=[12, 5])
        for i, ax in enumerate(axes):
            sy, sx = f[i].data.shape
            ax.imshow(f[i].data, cmap='afmhot') 
    fig.suptitle('raw '+fname)
    plt.show()


if True:
    sig = 1.5
    imf = gaussian_filter(f[0].data, sigma=sig, mode='mirror', order=0) # filter/smooth

if True:
    fig, ax = plt.subplots(1, 1)
    ax.imshow(imf, cmap='gray')
    plt.show()

gy = np.array([1, 0, -1]) * np.array([1, 2, 1])[:, None] 
gx = gy.T.copy()

Gx, Gy = [convolve(imf, g) for g in (gx, gy)]

G = np.sqrt(Gx**2 + Gy**2)
Th = np.arctan2(Gy, Gx)
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
a, b = np.histogram(G.round().astype('int32'),
    bins=range(np.amin(G).round().astype('int32'), np.amax(G).round().astype('int32')))
plt.plot(b[:-1], a)
plt.show()

if True:
    fig, ax = plt.subplots(1, 1)
    ax.imshow(G>300, cmap='gray')
    plt.show()

thresholds = np.arange(900, 1300, 100)
answers = []
for G_threshold in thresholds:
    keep = G > G_threshold
    anglez = np.mod(np.degrees(Th[keep]), 360)
    a, b = np.histogram(anglez, bins=range(0, 362, 2))
        # gray_hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    answers.append(a)

if True:
    for a in answers:
        plt.plot(b[:-1], a)
        plt.xticks([0, 30, 90, 150, 210, 270, 330])
    plt.show()


#plt.imsave('imf.png', imf)

cv2.imwrite("imf.png", imf)
img = cv2.imread('imf.png')
cv2.imshow('image', img)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eroded_imgGrey = cv2.erode(imgGrey,(7,7), iterations=5)
_, thrash = cv2.threshold(eroded_imgGrey, 40, 128, cv2.THRESH_BINARY)
canny = cv2.Canny(eroded_imgGrey, 125, 175)
#contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


object_edges = []
triangles_vertex = []
triangles_contours = []

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.12*cv2.arcLength(contour, True), True)
    #cv2.drawContours(imgGrey, [approx], 0, (128, 128, 128), 3)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    object_edges.append(len(approx))
    #print(len(approx))
    if len(approx) == 3:
        triangles_vertex.append(approx)
        triangles_contours.append(contour)
        cv2.drawContours(imgGrey, [approx], 0, (128, 128, 256), 6)
print('number of triangles:', len(triangles_vertex))
cv2.imshow("Edge Detection", imgGrey)
cv2.waitKey(0)

#thresholds2 = np.arange(900, 1300, 100)


def triangeArea(x1,y1,x2,y2,x3,y3):
  Triange_Area = abs((0.5)*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
  return Triange_Area

def three_angles(x1,y1,x2,y2,x3,y3):
    angle1 = np.arctan2((y2-y1),(x2-x1))
    angle2 = np.arctan2((y3-y2),(x3-x2))
    angle3 = np.arctan2((y1-y3),(x1-x3))
    return (angle1, angle2, angle3)


Theta = []
Weighting_Factor = []
for vertex, tri_contour in zip(triangles_vertex, triangles_contours):
    for i in range(3):
        Theta.append(three_angles(*vertex.flatten())[i])
        weight = cv2.contourArea(tri_contour)*np.ones_like(np.array(three_angles(*vertex.flatten())).shape)
        Weighting_Factor.append(weight[:])


Weighting_Factor = np.array(Weighting_Factor)
answers2 = []


angle = np.mod(np.degrees(Theta), 360)
a2, b2 = np.histogram(angle.reshape(*Weighting_Factor.shape), bins=range(0, 362, 2), weights=Weighting_Factor)
answers2.append(a2)

if True:
    for a2 in answers2:
        plt.plot(b2[:-1], a2)
        plt.xticks([0, 30, 90, 150, 210, 270, 330])
    plt.show()
