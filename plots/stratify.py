import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# parameters

models = ["deep-speech", "fine-tune", "fine-tune-aug", "freeze", "freeze-aug", "re-init", "encoder-decoder"]
colours = ["maroon", "orangered", "orange", "green", "lime", "teal", "cyan"]

# data frame

df = pd.DataFrame(columns=["Model", "Commands", "Digits", "Radio Alphabet", "Common", "Uncommon"])
for model in models:
    row = []
    results = open("C:\\Users\\charl\\Desktop\\AWS\\Results\\word category\\language\\" + model + ".txt").readlines()[0:-1]
    # results = open("C:\\Users\\charl\\Desktop\\AWS\\Results\\word category\\acoustic\\" + model + ".txt").readlines()[0:-1]
    for result in results:
        row.append(float("".join(_ for _ in result if _ in ".1234567890")))
    df.loc[len(df.index)] = [model] + row
    
# plot

fig, axs = plt.subplots(1, 1, figsize=(5.5, 3.5))
parallel_coordinates(df, 'Model' , color=colours, linewidth=2.5)
axs.set(ylim=(0, 150))
axs.set_ylabel("WER (%)", size=12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, fancybox=True, shadow=False, framealpha=1) #
fig.tight_layout()