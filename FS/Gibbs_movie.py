import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from IPython.display import HTML

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size':18})
plt.rcParams['animation.html'] = 'jshtml'
plt.style.use('dark_background')
plt.rcParams['animation.embed_limit'] = 100

dx = 0.001
L = 10
x = np.arange(0, L+dx, dx)
n = len(x)
nquart = int(np.floor(n/4))

f = np.zeros_like(x)
f[nquart:3*nquart] = 1

A0 = np.sum(f * np.ones_like(x)) * 2/L
fFS = A0/2 * np.ones_like(f)

fig, ax = plt.subplots()
plt.plot(x, f, color='w', linewidth=2)
fFS_plot, = plt.plot([], [], color='c', linewidth=2)

all_fFs = np.zeros((len(fFS), 1024))
all_fFs[:,0] = fFS

for k in range(1, 1024):
    Ak = np.sum(f * np.cos(2*np.pi*k*x/L)) * dx * 2/L
    Bk = np.sum(f * np.sin(2*np.pi*k*x/L)) * dx * 2/L
    fFS = fFS + Ak*np.cos(2*k*np.pi*x/L) + Bk*np.sin(2*np.pi*k*x/L)
    all_fFs[:, k] = fFS

def init():
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.2, 1.2) 
    return fFS

def animate(iter):
    fFS_plot.set_data(x, all_fFs[:, iter])
    return fFS_plot

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1024, interval=50)
HTML(anim.to_jshtml())  
anim.save('animation.html', writer='html')