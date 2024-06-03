import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread

plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')


A = imread("WT/OIG.jpg", format='jpg')
B = np.mean(A,-1)     # convert RGB to grayscale

plt.figure()
plt.imshow(256-A)   #, cmap='gray_r')
plt.axis('off') 

plt.show()

Bt = np.fft.fft2(B)
Btsort = np.sort(np.abs(Bt.reshape(-1)))        #sort by magnitude

# Zero out all small coefficients and inverse transform
for keep in (0.1, 0.05, 0.01, 0.002):
    thresh = Btsort[int(np.floor((1-keep)*len(Btsort)))]
    ind = np.abs(Bt)>thresh
    Btlow = Bt * ind
    Alow = np.fft.ifft2(Btlow).real
    plt.figure()
    plt.imshow(256-Alow, cmap='gray')
    plt.axis('off')
    plt.title('Compressed Image: keep = ' + str(keep*100) + '%')
    plt.show()
    
    
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['figure.figsize'] = [6,6]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(1, np.shape(B)[1]+1), np.arange(1, np.shape(B)[0]+1))
ax.plot_surface(X[0::10, 0::10], Y[0::10, 0::10], 256-B[0::10, 0::10], cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.mouse_init()
ax.view_init(270, 270)
plt.show()
    
