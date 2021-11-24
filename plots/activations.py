# tanh

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 1000)

# tanh

y1 = np.tanh(x)
y2 = 1 - np.tanh(x)**2

# figure

fig, axs = plt.subplots(1, 2, figsize = (5, 2))

axs[0].plot(x, y1, color = "red")
axs[0].set_xlabel("x", size = 10)
axs[0].set_ylabel(r"y", rotation = 0, size = 10)
axs[0].set_yticks([-1, 0,  1])
axs[0].set_xticks([-4, -2, 0, 2, 4])

axs[1].plot(x, y2 ,color = "green")
axs[1].set_ylabel(r"$\frac{\partial y}{\partial x}$", rotation = 0, size = 10, labelpad = 10)
axs[1].set_xticks([-4, -2, 0, 2, 4])
axs[1].set_xlabel("x", size = 10)


fig.tight_layout(w_pad=0.5)
