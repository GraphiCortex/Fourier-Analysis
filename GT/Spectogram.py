import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')


dt = 0.001
t = np.arange(0,2,dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2*np.pi*t*(f0 + (f1-f0)*np.power(t,2)/(3*t1**2)))

fs = 1/dt
sd.play(2*x, fs)

plt.specgram(x, NFFT=128, Fs=fs, noverlap=120, cmap='jet_r')
plt.colorbar()
plt.show()


# More Complex
import librosa
y, sr = librosa.load('SM/beethoven.mp3')
sd.play(y, sr)
plt.specgram(y[0:1000000], NFFT=5000, Fs=sr, noverlap=400, cmap=plt.cm.gist_heat)
plt.colorbar()
plt.show()

print(sr)           # Sampling Rate
print(np.size(y))

