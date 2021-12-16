# language weighting optimization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parameters

models = ["fine-tune", "fine-tune-aug", "freeze", "freeze-aug", "re-init"]
values = ["0", "25", "50", "75", "100"]
colours = ["orangered", "orange", "green", "lime", "teal"]

# obtain data

data = []
for model in models:
    sub_data = []
    for value in values:
        x = pd.read_csv("...\\lm-search\\" + model  + "\\results_" + value + ".csv") # insert path
        sub_data.append(x.iloc[0,5])
    data.append(sub_data)

# plot

fig, axs = plt.subplots(1, 1, figsize=(5, 3.2))
for i, sub_data in enumerate(data):
    axs.plot(np.arange(0, 1.25, 0.25), sub_data, label=models[i], marker="o", color=colours[i], alpha=0.9)
axs.set(ylim=(50, 110), xticks=np.arange(0, 1.25, 0.25), yticks=np.arange(50, 110, 10))
axs.set_xlabel(chr(945), size=12)
axs.set_ylabel("WER (%)", size=12)
# axs.legend(ncol=3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False, framealpha=1) #
fig.tight_layout()

    
