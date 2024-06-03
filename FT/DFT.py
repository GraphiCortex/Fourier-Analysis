import numpy as np
import matplotlib.pyplot as plt

n=256
I = 0 + 1j
J, K = np.meshgrid(np.arange(n), np.arange(n))
w = np.exp(-2*np.multiply(np.pi, I/n))
DFT = np.power(w, J*K)

DFT = np.real(DFT)
plt.imshow(DFT)
plt.show()
