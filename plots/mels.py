# mel filters

import numpy as np
import matplotlib.pyplot as plt

frameLength = 1000
nFilters = 16
sampleRate = 16000

low_freq_mel  = 0
high_freq_mel = 2595*np.log10(1 + sampleRate/(2*700))

mel_points = np.linspace(low_freq_mel, high_freq_mel, nFilters + 2)
hz_points  = 700*(10**(mel_points/2595) - 1)

bins = np.floor(frameLength*(hz_points/sampleRate))
filterBank = np.zeros((nFilters, int(frameLength/2)))

for m in range(1, nFilters + 1):
    f_m_minus = int(bins[m - 1])   # left
    f_m       = int(bins[m])       # center
    f_m_plus  = int(bins[m + 1])   # right
  
    for k in range(f_m_minus, f_m): # k == bin
        filterBank[m-1, k] = (k - bins[m-1])/(bins[m] - bins[m-1])
    for k in range(f_m, f_m_plus):
        filterBank[m-1, k] = (bins[m+1] - k)/(bins[m+1] - bins[m])

filterBank[:,:]

filterBank.shape

fig, axs = plt.subplots(1, 1, figsize = (5, 3))

for i in range(nFilters):
    axs.plot(np.linspace(0, sampleRate/2, frameLength//2), filterBank[i,:])
axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Amplitude")
fig.tight_layout()

