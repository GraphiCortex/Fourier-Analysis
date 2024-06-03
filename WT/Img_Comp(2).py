from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import pywt
plt.rcParams['figure.figsize'] = [16,16]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')

A = imread("WT/OIG.jpg", format='jpg')
B = np.mean(A,-1)   

# Waveelt Decomposition (2 level)
n = 2
w = 'dbl'           # Mother Wavelet 
coeffs = pywt.wavedec2(B, wavelet=w, level=n)

# Normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

plt.imshow(arr, cmap='gray_r', vmin=-0.25, vmax=0.75)
plt.rcParams['figure.figsize'] = [16,16]
fig=plt.figure(figsize=(18,16))
plt.show()

# Wavelet Compression
n = 4
w = 'db1'
coeffs = pywt.wavedec2(img, wavelet=w, level=n)

# convert coeffs to array
coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

Csort = np.sort(np.abs(coeffs_arr.reshape(-1)))

for keep in (0.01, 0.05, 0.1, 0.005):
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeffs_arr) > thresh
    Cfilt = coeffs_arr * ind # Threshold small indices

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

# Plot reconstruction
Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
plt.figure()
plt.imshow(Arecon.astype('uint8'), cmap='gray_r')
plt.axis('off')
plt.show()