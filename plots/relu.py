# relu

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

x = np.linspace(-4, 4, 1000)
# relu

y1 = []
x2 = []; y2 = []
x3 = []; y3 = []
for i in range(len(x)):
    if x[i] < 0:
        y1.append(0)
        x2.append(x[i])
        y2.append(0)
    else:
        y1.append(x[i])
        x3.append(x[i])
        y3.append(1)

# figure

fig, axs = plt.subplots(1, 2, figsize = (5, 2))

axs[0].plot(x, y1, color = "red")
axs[0].set_xlabel("x", size = 10)
axs[0].set_ylabel(r"y", rotation = 0, size = 10)
axs[0].set_yticks([0, 1, 2, 3, 4])
axs[0].set_xticks([-4, -2, 0, 2, 4])

axs[1].plot(x2, y2 ,color = "green")
axs[1].plot(x3, y3 ,color = "green")
axs[1].set_xlabel("x", size = 10)
axs[1].set_ylabel(r"$\frac{\partial y}{\partial x}$", rotation = 0, size = 10, labelpad = 10)
axs[1].set_xticks([-4, -2, 0, 2, 4])
# axs[1].set_xlabel("x", size = 10)

axs[1].add_artist(Ellipse((0, 1), 0.2, 0.03, color = "green"))
axs[1].add_artist(Ellipse((0.1, 0), 0.2, 0.03, color = "green", fill = False))

fig.tight_layout(w_pad=0.5)