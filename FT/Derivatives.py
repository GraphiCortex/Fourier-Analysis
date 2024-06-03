import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12,12]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')

n = 64
L = 30
dx = L/n
x = np.arange(-L/2, L/2, dx, dtype='complex_')
f = np.cos(x) * np.exp(-np.power(x,2)/25)
df = -(np.sin(x) * np.exp(-np.power(x,2)/25)) - (2/25)*x*f

# Approx der using finite diff
dfFD = np.zeros(len(df), dtype='complex_')
for kappa in range(len(df)-1):
    dfFD[kappa] = (f[kappa+1]-f[kappa])/dx

dfFD[-1]=dfFD[-2]

# Derivative using FFT (spectral der)
fhat = np.fft.fft(f)
kappa = (2*np.pi/L)*np.arange(-n/2,n/2)
kappa = np.fft.fftshift(kappa)      #Reorder fft freqs
dfhat = kappa * fhat * (1j)
dfFFT = np.real(np.fft.ifft(dfhat))

# Plots
plt.plot(x, df.real, color='w', linewidth=2, label='True Derivative')
plt.plot(x, dfFD.real, '--', color='r', linewidth=2, label='Finite Derivative')
plt.plot(x, dfFFT.real, '--', color='c', linewidth=2, label='FFT Derivative')
plt.legend()

plt.show()

