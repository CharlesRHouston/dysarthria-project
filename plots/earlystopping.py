# early stopping

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 50, 1000)

y1 = 5/x
y2 = 5/x + 0.001*(x-20)**2 + 0.6

fig, axs = plt.subplots(1, 1, figsize = (5, 3))

axs.plot(x, y1, color = "red", label = "Training error")
axs.plot(x, y2, color = "green", label = "Validation error")
axs.set_xlabel("Epochs", size = 12, labelpad = 8)
axs.set_ylabel("Loss", size = 12,  labelpad = 8)
axs.set_xticks([])
axs.set_yticks([])
axs.arrow(25, -0.1, 0, 0.9, head_width = 0.5, length_includes_head = True, head_length = 0.2,  color = "black")
axs.set_ylim((-0.1, 6))
axs.text(17, 1.2, "Early stopping point")
axs.legend()
