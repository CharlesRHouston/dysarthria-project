# augmentation
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
NFFT = 512
HOP_LENGTH = 320
POWER = 2.0
N_MELS = 64

# read in audio
fileName = r"C:\Users\charl\Desktop\Thesis\Data\UASpeech\UASpeech-3\M10\M10_B3_UW25_M7.wav"
rawSignal, sampleRate = librosa.load(fileName, sr = None)

# normalization
signal = rawSignal/np.max(rawSignal)

# cropping
signal = signal[10000:26000]

# generate log mel spectrogram
melSpectrogram = librosa.feature.melspectrogram(signal, sampleRate, n_fft=NFFT, hop_length=HOP_LENGTH, window=np.hamming, power=POWER, n_mels=N_MELS, center=False)
logMelSpectrogram = librosa.power_to_db(melSpectrogram)

# normalize to zero
logMelSpectrogram = logMelSpectrogram - np.mean(logMelSpectrogram)

# time mask

timeMask = np.copy(logMelSpectrogram)
T = logMelSpectrogram.shape[1]
a = 2
b = 5

for i in range(2):
    t = np.random.randint(a, b+1)
    t0 = np.random.randint(0, T+1-t)
    t1 = t0 + t
    timeMask[:,t0:t1] = 0

# frequency mask

frequencyMask = np.copy(logMelSpectrogram)
F = logMelSpectrogram.shape[0]
a = 3
b = 6

for i in range(2):
    f = np.random.randint(a, b+1)
    f0 = np.random.randint(0, F+1-f)
    f1 = f0 + f
    frequencyMask[f0:f1,:] = 0

# plot original
fig, axs = plt.subplots(3, 1, figsize = (4.5, 4.5), sharex=True)
img1 = librosa.display.specshow(logMelSpectrogram, sr=sampleRate, y_axis="mel", x_axis="s", hop_length=HOP_LENGTH, cmap="viridis", fmax=8192, ax=axs[0])
axs[0].set(xlabel="", title="Original")

# plot frequency
img2 = librosa.display.specshow(timeMask, sr=sampleRate, y_axis="mel", x_axis="s", hop_length=HOP_LENGTH, cmap="viridis", fmax=8192, ax=axs[1])
axs[1].set(xlabel="", title="Time mask")

# plot time
img3 = librosa.display.specshow(frequencyMask, sr=sampleRate, y_axis="mel", x_axis="s", hop_length=HOP_LENGTH, cmap="viridis", fmax=8192, ax=axs[2])
axs[2].set(title="Frequency mask")
# plot log mel spectrogram
# axs[1].set_xticks([0, 0.3, 0.6, 0.9])

fig.tight_layout()



