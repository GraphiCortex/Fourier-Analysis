import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm 

plt.rcParams['figure.figsize'] = [12,12]
plt.rcParams.update({'font.size':18})
plt.style.use('dark_background')

nu = 0.001  # Wave Speed
L = 20      # Length of Domain
N = 1000    # Number of discretization points
dx = L/N
x = np.arange(-L/2, L/2, dx)        # Define x domain

# Define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Initial condition
u0 = 1/np.cosh(x)

# Stimulate PDE in spatial domain
dt = 0.025
t = np.arange(0, 100*dt, dt)

def rhsBurgers(u, t, kappa, nu):
    uhat = np.fft.fft(u)
    d_uhat = (1j)*kappa*uhat
    dd_uhat = -np.power(kappa, 2) * uhat
    d_u = np.fft.ifft(d_uhat).real
    dd_u = np.fft.ifft(dd_uhat).real
    du_dt = -u * d_u + nu * dd_u
    return du_dt

u = odeint(rhsBurgers, u0, t, args=(kappa, nu))

# Waterfall Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u_plot = u[0:-1:10, :]
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    ax.plot(x, ys, u_plot[j,:], color=cm.jet(j*20))

# Image Plot
plt.figure()
plt.imshow(np.flipud(u), aspect=8)
plt.axis('off')
plt.set_cmap('jet_r')
plt.show()
