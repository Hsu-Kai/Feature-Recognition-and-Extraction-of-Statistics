import skimage.io
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy.ndimage.measurements import center_of_mass
import scipy.optimize as opt
from scipy import interpolate

image = skimage.io.imread(fname="Beam Energy 92.5.jpg")
skimage.io.imsave(fname="Beam Energy 92.5.png", arr=image)
image = skimage.io.imread(fname="Beam Energy 92.5.png")


# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")
plt.show()


imagec = skimage.io.imread(fname="Beam Energy 92.5.png")
imagec[imagec < 48] = 0
skimage.io.imsave(fname="contrast_adjusted Beam Energy 92.5.png", arr=imagec)


green=image[:, :, 1]
greenc=imagec[:, :, 1]

sig = 5
greenf = gaussian_filter(green, sigma=sig, mode='mirror', order=0) # filter/smooth
greencf = gaussian_filter(greenc, sigma=sig, mode='mirror', order=0) # contrast/filter/smooth


skimage.io.imsave(fname="Green_Beam Energy 92.5.png", arr=green)
skimage.io.imsave(fname="Green_filtered Beam Energy 92.5.png", arr=greenf)
skimage.io.imsave(fname="Green_filtered_contrast_adjusted Beam Energy 92.5.png", arr=greencf)

image = skimage.io.imread(fname="Green_filtered Beam Energy 92.5.png", as_gray=True)
imagec = skimage.io.imread(fname="contrast_adjusted Beam Energy 92.5.png", as_gray=True)



blob_log = blob_log(greenc, min_sigma=3.0, max_sigma=11.5, 
num_sigma=10, threshold=.095, overlap=0.0001)

yb, xb, sizes = blob_log.T.copy()

fig, ax = plt.subplots(1, 1)

ax.imshow(greencf, cmap='gray')
ax.plot(xb, yb, 'o', color='none', markeredgecolor='red', 
	markersize=14)
annoff=40
fs=14
rot=0
for i, (x, y, sizes) in enumerate(zip(xb, yb, sizes)):
    ax.annotate(str(i), [x+annoff, y], color='r'
		,fontsize=fs, rotation=rot)
    print(i,(y, x, sizes))
plt.show()

BLGlocations = np.array([[521, 2017], [209, 1371], [1630, 1468], [1322, 836],
                      [622, 776], [1225, 2050]]) #0,2,3,4,5,15 imagec[imagec < 48] = 0

hw = 8
BLGspots = []
for BLGy, BLGx in BLGlocations:
    mask1 = np.ones_like(greencf, dtype=bool)
    mask1[BLGy-hw:BLGy+hw+1, BLGx-hw:BLGx+hw+1] = False
    ma1 = np.ma.array(greencf, mask=mask1)
    BLGspots.append(center_of_mass(ma1))

BLGspots = np.array(BLGspots)

center = BLGspots.mean(axis=0)
yc, xc = center

r_BLGspots = np.sqrt(((BLGspots - center)**2).sum(axis=1))
print('center: ', center, 'r_BLGspots: ', r_BLGspots)


def sum_distance(center_opt):
    sum = 0
    sum = sum + np.sqrt(((BLGspots - center_opt)**2).sum(axis=1)).sum(axis=0)
    return sum

result = opt.minimize(sum_distance, x0 = [922, 1420], method= 'Nelder-Mead', tol=1e-6)
print ('result:', result)

center_opt = [926.95795439, 1419.48802291]
r_opt = np.sqrt(((BLGspots - center)**2).sum(axis=1))
print('center_opt: ', center_opt, 'r_opt: ', r_opt)

yc_opt, xc_opt = center_opt

TaS2locations = np.array([[1451, 1450], [706, 935], [397, 1370], [1187, 1885]]) #6,7,9,11 imagec[imagec < 48] = 0

TaS2spots = []
for TaS2y, TaS2x in TaS2locations:
    mask2 = np.ones_like(greencf, dtype=bool)
    mask2[TaS2y-hw:TaS2y+hw+1, TaS2x-hw:TaS2x+hw+1] = False
    ma2 = np.ma.array(greencf, mask=mask2)
    TaS2spots.append(center_of_mass(ma2))

TaS2spots = np.array(TaS2spots)

r_TaS2spots = np.sqrt(((TaS2spots - center)**2).sum(axis=1))
print('r_TaS2spots: ', r_TaS2spots)


rmin, rmax = 495, 555

theta = np.linspace(0, 2*np.pi, 1*360+1)
r = np.arange(495, 555) #rmin, rmax = 527, 536
y_r, x_theta = [r[:, None] * f(theta) +c for f, c in zip((np.cos, np.sin), center_opt)]

coords = np.vstack((y_r.flatten(), x_theta.flatten()))
order_resample = 1 # first order interpolation (up to 3)
const = 0.
rotational_spectrum = map_coordinates(greenf, coords, mode='constant',
                                 order=order_resample,
                                 cval=const).reshape(*x_theta.shape)


th = np.linspace(0, 2*np.pi, 201)
s, c = [f(th) for f in (np.sin, np.cos)]
   
fig = plt.figure(figsize=[10.5, 15])
    
plt.subplot(2, 1, 1)
plt.imshow(greenf, cmap='gray')
y, x = BLGspots.T
plt.plot(x, y, 'o', color='none', markeredgecolor='red', markersize=15)
plt.plot(xc_opt, yc_opt, 'or', markersize=12)
plt.plot(c*rmin+xc_opt, s*rmin+yc_opt, '-r')
plt.plot(c*rmax+xc_opt, s*rmax+yc_opt, '-r')
plt.subplot(2, 1, 2)
plt.imshow(rotational_spectrum, cmap='gray', origin='lower')
    
thing = rotational_spectrum[:].sum(axis=0)
plt.plot(0.012*(thing.astype(np.float)), '-c')
plt.xlim(0, 360)
plt.xticks([0, 30, 90, 150, 210, 270, 330])
plt.show()



#Checking position-dependent background at BLG spots
hw2 = 50
fig = plt.figure(figsize=[10.5, 15])
plt.subplot(2, 1, 1)
plt.imshow(greenf, cmap='gray')
y, x = BLGspots.T
plt.plot(x, y, 'o', color='none', markeredgecolor='red', markersize=15)
plt.plot(xc_opt, yc_opt, 'or', markersize=12)

for i, (BLGy, BLGx) in enumerate(BLGlocations):
    plt.subplot(2, 6, i+7)
    cropi = image[BLGy-hw2:BLGy+hw2, BLGx-hw2:BLGx+hw2]
    plt.imshow(cropi, cmap='gray')
    h_mid = cropi[:, 40:61].sum(axis=1)
    v_mid = cropi[40:61,:].sum(axis=0)
    plt.plot(range(100), 100-0.03*h_mid.astype(np.float), '-r')
    plt.plot(0.03*v_mid.astype(np.float), range(100), '-g')
plt.show()



# Define Gaussian Function for Fitting 
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


#Generate Data For Fitting BLG spots
x = np.linspace(0, 99, 100)                                # Create x and y indices
y = np.linspace(0, 99, 100)
x, y = np.meshgrid(x, y)
xdata_tuple = (x,y)
data = twoD_Gaussian(xdata_tuple, 3, 50, 50, 20, 20, 0, 5) #create data & trial parameters:(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)

plt.figure()                                               # plot twoD_Gaussian data generated above
plt.imshow(data.reshape(100, 100))
plt.colorbar()
plt.show()



#Gaussian Fitting for BLG spots
for i, (BLGy, BLGx) in enumerate(BLGlocations):
    plt.subplot(2, 6, i+1)
    cropi = image[BLGy-hw2:BLGy+hw2, BLGx-hw2:BLGx+hw2]
    plt.imshow(cropi, cmap='gray', origin='lower')
    plt.subplot(2, 6, i+7)
    initial_guess = (3, 50, 50, 20, 20, 0, 5)
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata_tuple, cropi.reshape(10000), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata_tuple, *popt)
    plt.imshow(cropi.reshape(100, 100), cmap=plt.cm.jet, origin='lower')
    plt.contour(x, y, data_fitted.reshape(100, 100), 8, colors='w', origin='lower')
plt.show()





#Generate Data For Fitting TaS2
x = np.linspace(0, 29, 30)                                # Create x and y indices
y = np.linspace(0, 59, 60)
x, y = np.meshgrid(x, y)
xdata_tuple = (x,y)
data = twoD_Gaussian(xdata_tuple, 40, 20, 30, 4.5, 20, 0, 20) #create data & trial parameters:(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
plt.figure()                                               # plot twoD_Gaussian data generated above
plt.imshow(data.reshape(60, 30))
plt.colorbar()
plt.show()


#Gaussian Fitting for TaS2
a_0 = [np.hstack([rotational_spectrum[:,345:360],rotational_spectrum[:,0:15]]), rotational_spectrum[:, 45:75],
       rotational_spectrum[:, 105:135], rotational_spectrum[:, 165:195],
       rotational_spectrum[:, 225:255], rotational_spectrum[:, 285:315]]
l_0 = ["0 (+-15deg)", "60 (+-15deg)", "120 (+-15deg)", "180 (+-15deg)", "240 (+-15deg)", "300 (+-15deg)"]
a_30 = [rotational_spectrum[:, 15:45], rotational_spectrum[:, 75:105],
        rotational_spectrum[:, 135:165], rotational_spectrum[:, 195:225],
        rotational_spectrum[:, 255:285], rotational_spectrum[:, 315:345]]
l_30 = ["30 (+-15deg)", "90 (+-15deg)", "150 (+-15deg)", "210 (+-15deg)", "270 (+-15deg)", "330 (+-15deg)"]

a_0_opt = []
for i, (crop_a0, l_0)  in enumerate(zip(a_0, l_0)):
    plt.subplot(2, 6, i+1)
    plt.imshow(crop_a0, cmap='gray', origin='lower')
    plt.xticks([])
    plt.xlabel(l_0)    
    plt.subplot(2, 6, i+7)
    initial_guess = (40, 20, 30, 4.5, 20, 0, 20)
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata_tuple, crop_a0.reshape(1800), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata_tuple, *popt)
    print(popt)
    [x1,x2]=crop_a0.shape
    X1, X2 = np.mgrid[:x1, :x2]
    X = np.hstack(   ( np.reshape(X1, (x1*x2, 1)) , np.reshape(X2, (x1*x2, 1)) ) )
    X = np.hstack(   ( np.ones((x1*x2, 1)) , X ))
    YY = np.reshape(data_fitted.reshape(60, 30), (x1*x2, 1))
    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (x1, x2))
    plane_shift = plane - np.ones_like(plane, dtype=np.float64)*(np.amax(plane)-np.amin(data_fitted))
    
    Y_sub = data_fitted.reshape(60, 30) - plane_shift
    a_0_opt.append(Y_sub)
    plt.imshow(Y_sub, cmap=plt.cm.jet, origin='lower')
    plt.xticks([])
    plt.xlabel(l_0) 
plt.show()


a_30_opt = []
for j, (crop_a30, l_30)  in enumerate(zip(a_30, l_30)):
    plt.subplot(2, 6, j+1)
    plt.imshow(crop_a30, cmap='gray', origin='lower')
    plt.xticks([])
    plt.xlabel(l_30) 
    plt.subplot(2, 6, j+7)
    initial_guess = (10, 17, 30, 6, 20, 0, 20)
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata_tuple, crop_a30.reshape(1800), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata_tuple, *popt)
    print(popt)
    [x1,x2]=crop_a30.shape
    X1, X2 = np.mgrid[:x1, :x2]
    X = np.hstack(   ( np.reshape(X1, (x1*x2, 1)) , np.reshape(X2, (x1*x2, 1)) ) )
    X = np.hstack(   ( np.ones((x1*x2, 1)) , X ))
    YY = np.reshape(data_fitted.reshape(60, 30), (x1*x2, 1))
    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (x1, x2))
    plane_shift = plane - np.ones_like(plane, dtype=np.float64)*(np.amax(plane)-np.amin(data_fitted))
    
    Y_sub = data_fitted.reshape(60, 30) - plane_shift
    a_30_opt.append(Y_sub)
    plt.imshow(Y_sub, cmap=plt.cm.jet, origin='lower')
    plt.xticks([])
    plt.xlabel(l_30) 
plt.show()

a_0_opt = np.array(a_0_opt)
a_30_opt = np.array(a_30_opt)


# remove the section in 0-degree domain blocked by LEED electron gun
# remove a near section in 30-degree domain for symmetry consideration
a_0_opt = np.array(a_0_opt)[[0,2,3,4,5],:,:]
a_30_opt = np.array(a_30_opt)[[0,2,3,4,5],:,:]
for i, crop_a0  in enumerate(a_0_opt):
    plt.subplot(2, 6, i+1)
    plt.xticks([])
    plt.imshow(crop_a0, cmap='gray', origin='lower')
for j, crop_a30  in enumerate(a_30_opt):
    plt.subplot(2, 6, j+7)
    plt.xticks([])
    plt.imshow(crop_a30, cmap='gray', origin='lower')    
plt.show()

intensity_0degree = a_0_opt.sum(axis=0).sum(axis=0).sum(axis=0)
intensity_30degree = a_30_opt.sum(axis=0).sum(axis=0).sum(axis=0)
print('0 degree percentage: ', (intensity_0degree)/(intensity_0degree+intensity_30degree))   #0 degree percentage:  0.7883570134151674
print('30 degree percentage: ', (intensity_30degree)/(intensity_0degree+intensity_30degree)) #30 degree percentage:  0.2116429865848327
