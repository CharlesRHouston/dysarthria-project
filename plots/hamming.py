# hamming function

import matplotlib.pyplot as plt
import numpy as np

N = 100
x = np.linspace(0, N, N+1)
y = 0.54 - 0.46*np.cos(2*np.pi*x/(N-1))

fig, axs = plt.subplots(1, 1, figsize = (2.5, 2))
axs.plot(x, y, "r-")
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.set_ylim(0, 1.05)
axs.set_xticks(np.arange(0, 125, 25))
axs.set_yticks(np.arange(0, 1.25, 0.25))
fig.tight_layout()