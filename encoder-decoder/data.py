# extract features from raw waveforms, stipulate masks and generate one-hot targets 

# modules
import pandas
import librosa
import numpy as np 
import tensorflow as tf

# create data tensors using the csv information (path, size, transcript) 
def feature_extraction(filepath, NFFT, HOP, POWER, MELS, N_CHARS, CHARACTERS, MFCC):
    # read csv 
    df = pandas.read_csv(filepath)
    
    # declare storage
    X = []
    Y = []
    X_mask = []
    Y_mask = []
    
    #  iterate through examples
    for i, file in enumerate(df["wav_filename"]):
        
        ### input ###
        
        # read in wav file
        signal, sampleRate = librosa.load(file, sr=None)
        # normalize signal
        signal = signal/max(signal)
        
        # mel spectrogram
        # melSpectrogram = librosa.feature.melspectrogram(signal, sampleRate, n_fft=NFFT, hop_length=HOP, window=np.hamming, power=POWER, n_mels=MELS, center=False)
        # log mel spectrogram
        # logMelSpectrogram = librosa.power_to_db(melSpectrogram)
        
        # mfcc
        logMelSpectrogram = librosa.feature.mfcc(signal, sampleRate, n_mfcc=MFCC)
        
        # rotate so that dimensions are (timesteps, features)
        x = np.rot90(logMelSpectrogram)
        
        ### target ###
        
        # convert from string to one-hot array
        target = df["transcript"][i]
        y = np.zeros((len(target)+2, N_CHARS)) # (timesteps, features)
        # start token
        y[0, CHARACTERS.index('<S>')] = 1   
        # characters
        for i, char in enumerate(target):
            y[i+1, CHARACTERS.index(char)] = 1
        # end token
        y[-1, CHARACTERS.index('<E>')] = 1
        
        # append to batch
        X.append(x)
        Y.append(y)
        
        ### mask ###
        
        X_mask.append(np.ones((x.shape[0])))
        Y_mask.append(np.ones((y.shape[0]-1)))
        
    # padding
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')
    Y = tf.keras.preprocessing.sequence.pad_sequences(Y, padding='post')
    X_mask = tf.keras.preprocessing.sequence.pad_sequences(X_mask, padding='post')
    Y_mask = tf.keras.preprocessing.sequence.pad_sequences(Y_mask, padding='post')
    
    # convert masks to boolean
    X_mask = X_mask == 1 
    Y_mask = Y_mask == 1 
    
    return X, Y[:,0:-1,:], Y[:,1:,:], X_mask, Y_mask

