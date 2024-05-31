import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')

#Fourier for discontinuous functions (Gibb's phenomena)

dx = 0.001
L = 2*np.pi
x = np.arange(0, L+dx, dx)
n = len(x)
nquart = int(np.floor(n/4))

f = np.zeros_like(x)
f[nquart:3*nquart] = 1

A0 = np.sum(f * np.ones_like(x)) * dx * 2/L
fFS = A0/2 * np.ones_like(f)

for k in range(1, 1024):
    Ak = np.sum(f * np.cos(np.pi*2*k*x/L)) * dx *2/L
    Bk = np.sum(f * np.sin(np.pi*2*k*x/L)) * dx *2/L
    fFS = fFS + Ak*np.cos(np.pi*2*k*x/L) + Bk * np.sin(np.pi*2*k*x/L)

plt.plot(x, f, color='w', linewidth=2)
plt.plot(x, fFS, '-', color='c', linewidth=1.5)
plt.show()