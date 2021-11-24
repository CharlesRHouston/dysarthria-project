# intelligibility results

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parameters

models = ["deep-speech", "fine-tune", "fine-tune-aug", "freeze", "freeze-aug", "re-init", "encoder-decoder"]
values = ["Very Low", "Low", "Mid", "High"]
colours = ["maroon", "orangered", "orange", "green", "lime", "teal", "cyan"]

# obtain data

data = []
for model in models[0:-1]:
    x = pd.read_csv("C:\\Users\\charl\\Desktop\\AWS\\Results\\deep-speech\\language\\" + model  + "\\results.csv")
    data.append(np.array(x.iloc[0,1:-1], dtype=np.float32))
data.append(np.array([97.25, 93.99, 93.83, 90.2]))

# plot

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
width = 0.12
x_axis = np.arange(0, 4, 1)
for i, model in enumerate(models):
    x_axis = [x + width for x in x_axis]
    axs.bar(x_axis , data[i], label=model, width=width, align="center", alpha=0.7, color = colours[i])

axs.set(xticks= [x + 4*width for x in np.arange(0, 4, 1)], xticklabels=values, ylim=(0, 180), yticks=np.arange(0, 200, 20)) 
axs.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=False, framealpha=1) #
axs.set_xlabel("Intelligibility", size=12)
axs.set_ylabel("WER (%)", size=12)
fig.tight_layout()



