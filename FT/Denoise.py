import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16,12]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')


# Create a simple signal with two frequencies
dt = 0.001
t = np.arange(0, 1, dt)
f = np.sin(2*np.pi*50*t) + np.cos(2*np.pi*120*t)    #Sum of two frequencies   
f_clean = f
f = f + 2.5*np.random.randn(len(t))                 #adding some noise

plt.plot(t, f, color='c', linewidth=2, label='Noisy')
plt.plot(t, f_clean, color='w', linewidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.legend()
plt.show()



#Compute the fast fourier transform (FFT)
n = len(t)
fhat = np.fft.fft(f, n)                             # compute the fft
PSD = fhat * np.conj(fhat) / n                      # Power Spectrum (since lambda * lambda.conj gives us |lambda|^2)
freq = (1/dt*n) * np.arange(n)                      # Create x-axis of frequencies
L = np.arange(1, np.floor(n/2), dtype='int')        

fig, axs = plt.subplots(2, 1)

plt.sca(axs[0])
plt.plot(t, f, color='c', linewidth=2, label='Noisy')
plt.plot(t, f_clean, color='w', linewidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.legend()


plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='c', linewidth=2, label='Noisy')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()


# Use PSD to filter out noise
indices = PSD > 100             # Find all freqs with large power
PSDclean = PSD * indices        # Zero out all other
fhat = indices * fhat           # Zero out small Fourier Coeffs, in Y
ffilt = np.fft.ifft(fhat)       # Inverse FFT for filtered time signal

# Plots
fig, axs = plt.subplots(4, 1)

plt.sca(axs[0])
plt.plot(t, f, color='c', linewidth=2, label='Noisy')
plt.plot(t, f_clean, color='w', linewidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[2])
plt.plot(t, ffilt, color='w', linewidth=2, label='Filtered')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='c', linewidth=2, label='Noisy')
plt.plot(freq[L], PSDclean[L], color='w', linewidth=2, label='Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.sca(axs[3])
plt.plot(t, f_clean, color='w', linewidth=2, label='Clean')
plt.plot(t, ffilt, color='c', linewidth=2, label='Filtered')
plt.xlim(t[0], t[-1])
plt.legend()

plt.show()

