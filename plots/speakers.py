# speakers box-plots

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# parameters

models = ["deep-speech", "fine-tune", "fine-tune-aug", "freeze", "freeze-aug", "re-init", "encoder-decoder"]
colours = ["maroon", "orangered", "orange", "green", "lime", "teal", "cyan"]
cmap = sns.color_palette(colours)

data = []
for model in models[0:-1]:
    x = pd.read_csv("C:\\Users\\charl\\Desktop\\AWS\\Results\\deep-speech\\language\\" + model  + "\\speakers.csv")
    data.append(x.iloc[0,:])
data.append(np.array([91.37, 97.24, 94.44, 89.8, 95.69, 98.04, 93.33, 94.9, 90.2, 92.16, 90.2, 93.73, 98.04, 88.63, 95.69]))

# pyplot boxplot

# fig, axs = plt.subplots(1, 1, figsize=(7, 4))
# axs.boxplot(data, labels=models)
# axs.set(xlabel="Model", ylabel="Speaker WER (%)")
# fig.tight_layout()

# seaborn boxplot

fig, axs = plt.subplots(1, 1, figsize=(5.5, 3.2))
# sns.set_style(style="white",rc= {'patch.edgecolor': 'black'})
sns.boxplot(data=data, color="white", linewidth=1.5, width=0.7)
sns.stripplot(data=data, color="black", size=4, alpha=1.0)
axs.set_xlabel("Models", fontsize=12)
axs.set_ylabel("Speaker WER (%)", fontsize=12)
axs.set(xticklabels=models, ylim=(0, 170))
axs.tick_params(axis='both', which='major', labelsize=8)
plt.xticks(rotation=20)
fig.tight_layout()