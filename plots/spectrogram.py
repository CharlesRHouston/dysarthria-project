# librosa log mel spectrogram
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
NFFT = 512
HOP_LENGTH = 320
POWER = 2.0
N_MELS = 64
MFCC = 13

# read in audio
fileName = r"...\UASpeech\UASpeech-3\M10\M10_B3_UW25_M7.wav" # insert path
rawSignal, sampleRate = librosa.load(fileName, sr = None)

# normalization
signal = rawSignal/np.max(rawSignal)

# cropping
signal = signal[10000:26000]

# generate log mel spectrogram
melSpectrogram = librosa.feature.melspectrogram(signal, sampleRate, n_fft=NFFT, hop_length=HOP_LENGTH, window=np.hamming, power=POWER, n_mels=N_MELS, center=False)
logMelSpectrogram = librosa.power_to_db(melSpectrogram)

# generate mfcc
mfcc = librosa.feature.mfcc(S=logMelSpectrogram, n_mfcc=MFCC, dct_type=2, norm='ortho')

# plot signal
fig, axs = plt.subplots(1, 2, figsize = (6, 2))
axs[0].plot(np.linspace(0, len(signal)/sampleRate, len(signal)), signal, linewidth=0.5, color="black")
axs[0].set_xlabel("Time (s)", size=12)
axs[0].set_ylabel("Pressure", size=12)
axs[0].set_xticks([0, 0.3, 0.6, 0.9])

# plot log mel spectrogram
librosa.display.specshow(logMelSpectrogram, sr=sampleRate, y_axis="mel", x_axis="s", hop_length=HOP_LENGTH, cmap="viridis", fmax=8192)
# librosa.display.specshow(mfcc, sr=sampleRate, cmap="viridis", x_axis='time')
# axs[1].imshow(mfcc[2:13,:], cmap = 'viridis')
plt.colorbar()
axs[1].set_xticks([0, 0.3, 0.6, 0.9])
fig.tight_layout()


# plot log mel spectrogram (Deep Speech diagram)
# fig, axs = plt.subplots(figsize = (6, 1))
# img = librosa.display.specshow(logMelSpectrogram, sr=sampleRate, y_axis="mel", x_axis="s", hop_length=HOP_LENGTH, cmap="viridis", fmax=8192)
# axs.set(xlabel= "", ylabel = "", xticks = [], yticks = [])
