# UASpeech Plots

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os
import librosa
from curlyBrace import curlyBrace
import nltk

# 1. intelligibility plot

SPEECH = {"F02": ["Low", 29], "F03": ["Very Low", 6], "F04": ["Mid", 62], \
          "F05": ["High", 95], "M01": ["Very Low", 15], "M04": ["Very Low", 2], \
          "M05": ["Mid", 58], "M07": ["Low", 28], "M08": ["High", 93], \
          "M09": ["High", 86], "M10": ["High", 93], "M11": ["Mid", 62], \
          "M12": ["Very Low", 7.4], "M14": ["High", 90.4], "M16": ["Low", 43]}
sorted_speech = dict(sorted(SPEECH.items(), key = lambda x: x[1][1]))

LEVELS = ["Very Low", "Low", "Mid", "High"]

level_count = [0, 0, 0, 0]
for value in SPEECH.values():
    level_count[LEVELS.index(value[0])] += 1

fig, axs = plt.subplots(figsize = (8.4, 4.8))
axs.set_ylim((0, 100))
axs.set_xlabel("Speaker ID", size = 14)
axs.set_ylabel("Intelligibility (%)", size = 14)
axs.set_yticks(np.arange(0, 125, 25))

axs.hlines(25, -1, 15, color="black", zorder=2, linestyle="dashed", linewidth=0.8)
axs.hlines(50, -1, 15, color="black", zorder=2, linestyle="dashed", linewidth=0.8)
axs.hlines(75, -1, 15, color="black", zorder=2, linestyle="dashed", linewidth=0.8)

# axs.bar(sorted_speech.keys(), [val[1] for val in sorted_speech.values()], color = "grey", alpha = 1, edgecolor = "black", linewidth = 1.2, zorder=1)

axs.text(-0.5, 18, "Very Low", size = 12)
axs.text(-0.5, 43, "Low", size = 12)
axs.text(-0.5, 68, "Mid", size = 12)
axs.text(-0.5, 93, "High", size = 12)

(markers, stemlines, baseline) = axs.stem(sorted_speech.keys(), [val[1] for val in sorted_speech.values()])
plt.setp(stemlines, linestyle="-", color="grey", linewidth=1.5)
plt.setp(markers, color="black")
plt.setp(baseline, linewidth=0)

# 2. duration plot

durations = []
long = []
for path, direc, files in os.walk(r"..."): # insert path to UASpeech data
    if os.path.basename(path) not in SPEECH.keys():
        continue
    for file in files:
        sample_rate, signal = wavfile.read(path + "/" + file)
        duration = len(signal)/sample_rate
        durations.append(duration)
        if duration > 20:
            long.append(duration)
    
# fig, axs = plt.subplots(1, 1, figsize=(8, 4.5), gridspec_kw={'width_ratios': [2.5, 1]})
fig, axs = plt.subplots(1, 1, figsize=(7, 4.5))
axs.grid(zorder=0)
axs.hist(durations, bins = np.arange(0, 50, 1), align = "mid", range = (0, 50), edgecolor = "black",zorder = 3, color = ['firebrick'])
axs.set_xlim((0, 16))
axs.set_ylim((0, 6000))
axs.set_xticks(np.arange(0, 17, 1))
axs.set_xlabel("Duration (s)", size = 14)
axs.set_ylabel("Frequency", size = 14)
fig.tight_layout()

fig, axs = plt.subplots(1, 1, figsize=(3, 5))
res = axs.boxplot(durations, widths = 0.6, flierprops = dict(marker = 'o', markersize = 6), labels = [""])
axs.set_ylabel("Duration (s)", size = 14)
axs.set_xlabel("UASpeech", size = 14)
axs.set_ylim((0, 50))
# axs.spines['top'].set_visible(False)
# axs.spines['right'].set_visible(False)
# axs.spines['left'].set_visible(False)
axs.grid(color='black', axis='y', linestyle='-', linewidth=1, alpha=0.5)
fig.tight_layout()

[item.get_ydata() for item in res['whiskers']]
[item.get_ydata() for item in res['medians']]
[item.get_ydata() for item in res['boxes']]

# 3. waveform plot

# phone translation

allWordToPhonesDict = nltk.corpus.cmudict.dict()
print(allWordToPhonesDict["left"])
print(allWordToPhonesDict["house"])

print(allWordToPhonesDict["feed"])
print(allWordToPhonesDict["thorn"])

# left

# standard file names
leftStandard = r"...\SpeechCommands\left\00f0204f_nohash_0.wav" # insert path to Speech Commands data

# impaired file names
leftImpaired = r"...\UASpeech\F03\F03_B1_C18_M7.wav" # insert path to UASpeech data

# plot signal
fig, axes = plt.subplots(2, 1)

# standard
signal, sampleRate = librosa.load(leftStandard, sr = None)
signal = signal/max(signal)

axes[0].plot(np.linspace(0, 1, sampleRate), signal, linewidth = 0.3, color = "black")
axes[0].set_title("\"Left\" Standard Speech")
axes[0].set_ylim((-1.05, 2))
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Pressure")

curlyBrace(fig, axes[0], (300/sampleRate, 1.1), (3300/sampleRate, 1.1), color = "black", str_text = "[L]")
curlyBrace(fig, axes[0], (3700/sampleRate, 1.1), (6700/sampleRate, 1.1), color = "black", str_text = "[EH]")
curlyBrace(fig, axes[0], (7000/sampleRate, 1.1), (10800/sampleRate, 1.1), 0.08,  color = "black", str_text = "[F]")
curlyBrace(fig, axes[0], (11200/sampleRate, 1.1), (15000/sampleRate, 1.1), 0.08, color = "black", str_text = "[T]")

# impaired
signal, sampleRate = librosa.load(leftImpaired, sr = None)
signal = signal[28000:44000]
signal = signal/max(signal)

axes[1].plot(np.linspace(0, 1, sampleRate), signal, linewidth = 0.3, color = "black")
axes[1].set_title("\"Left\" Dysarthric Speech")
axes[1].set_ylim((-1.05, 2))
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Pressure")

curlyBrace(fig, axes[1], (300/sampleRate, 1.1), (5000/sampleRate, 1.1), 0.07, color = "black", str_text = "[L]")
curlyBrace(fig, axes[1], (5300/sampleRate, 1.1), (8800/sampleRate, 1.1), 0.08, color = "black", str_text = "[EH]")
curlyBrace(fig, axes[1], (9100/sampleRate, 1.1), (12000/sampleRate, 1.1),  color = "black", str_text = "[F]")

fig.tight_layout()

# HOUSE

# standard file names
houseStandard = r"...\SpeechCommands\house\00b01445_nohash_1.wav" # insert path to Speech Commands data

# impaired file names
houseImpaired = r"...\UASpeech\M04\M04_B1_UW82_M7.wav" # insert path to Speech Commands data

# plot signal
fig, axes = plt.subplots(2, 1)

# standard
signal, sampleRate = librosa.load(houseStandard, sr = None)
signal = signal/max(signal)
axes[0].plot(np.linspace(0, 1, sampleRate), signal, linewidth = 0.3, color = "black")
axes[0].set_title("\"House\" Standard Speech")
axes[0].set_ylim((-1.05, 2))
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Pressure")

curlyBrace(fig, axes[0], (300/sampleRate, 1.1), (4900/sampleRate, 1.1), 0.07, color = "black", str_text = "[HH]")
curlyBrace(fig, axes[0], (5200/sampleRate, 1.1), (8600/sampleRate, 1.1), 0.08, color = "black", str_text = "[AW]")
curlyBrace(fig, axes[0], (8900/sampleRate, 1.1), (14000/sampleRate, 1.1), 0.06, color = "black", str_text = "[S]")

# impaired
signal, sampleRate = librosa.load(houseImpaired, sr = None)
signal = signal[8000:24000]
signal = signal/max(signal)
axes[1].plot(np.linspace(0, 1, sampleRate), signal, linewidth = 0.3, color = "black")
axes[1].set_title("\"House\" Dysarthric Speech")
axes[1].set_ylim((-1.4, 2))
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Pressure")

curlyBrace(fig, axes[1], (5100/sampleRate, 1.1), (7700/sampleRate, 1.1), color = "black", str_text = "[AW]")
curlyBrace(fig, axes[1], (7900/sampleRate, 1.1), (11500/sampleRate, 1.1),  0.08, color = "black", str_text = "[S]")

fig.tight_layout()
