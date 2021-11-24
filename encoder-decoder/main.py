# Encoder-Decoder Recurrent Neural Network

# python version
import sys
sys.version # 3.7.10

# MODULES 

# standard modules
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas

# custom modules
from data import feature_extraction
from results import impaired_errors, standard_errors

################## Section 1: Training ##################

# HYPER-PARAMETERS 

# feature extraction hyper-parameters
NFFT    = 512       # 32 ms
HOP     = 320       # 20 ms
POWER   = 2.0       # power spectrogram
MELS    = 64        # Mel filters
MFCC    = 26        # number of MFCC coefficients

# modelling hyper-parameters
ENCODER     = 128   # hidden units
DECODER     = 128   # hidden units
EPOCHS      = 300   # number of passes through data, early stopping used
BATCH       = 32    # batch size
DELTA       = 0.05  # used in early stopping  
PATIENCE    = 20    # epochs before early stopping triggered

# characters: alphabet + space + start/end tokens
CHARACTERS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '<S>', '<E>')
N_CHARS = len(CHARACTERS)   # total number of characters

# impaired speaker codes and respective intelligibility levels
SPEAKERS = {"F02": "Low", "F03": "Very Low", "F04": "Mid", "F05": "High", "M01": "Very Low", "M04": "Very Low", "M05": "Mid", "M07": "Low", "M08": "High", "M09": "High", "M10": "High", "M11": "Mid", "M12": "Very Low", "M14": "High", "M16": "Low"}


# DATA 

# see 'data' module
# tensors of shape: (batch, timesteps, features)
# masks of shape:   (batch, timesteps)


# TRAINING

# define the sequence-to-sequence training model
def construct_model(encoder_shape, decoder_shape):
    # encoder
    encoder_input = Input(encoder_shape, name="encoder_input")
    encoder_mask = Input((encoder_shape[0],), dtype=tf.bool, name="encoder_mask")
    encoder_output, state_h, state_c = LSTM(units=ENCODER, return_state=True, name="encoder_lstm")(encoder_input, mask=encoder_mask)
    
    # carry forward hidden states
    encoder_states = [state_h, state_c]
    
    # decoder
    decoder_input = Input(decoder_shape, name="decoder_input")
    decoder_mask = Input((decoder_shape[0],), dtype=tf.bool, name="decoder_mask")
    decoder_output, _, _ = LSTM(units=DECODER, return_sequences=True, return_state=True, name="decoder_lstm")(decoder_input, initial_state=encoder_states, mask=decoder_mask)
    decoder_output = Dense(N_CHARS, activation='softmax', name="output_layer")(decoder_output)
    
    # model
    model = Model(inputs = [encoder_input, decoder_input, encoder_mask, decoder_mask], outputs = decoder_output)
    return model


# train the sequence-to-sequence model
def train_model(train_path, valid_path):
    # extract training and validation data
    x1, x2, y, x1_mask, x2_mask = feature_extraction(train_path, NFFT, HOP, POWER, MELS, N_CHARS, CHARACTERS, MFCC)
    x1_val, x2_val, y_val, x1_val_mask, x2_val_mask = feature_extraction(valid_path, NFFT, HOP, POWER, MELS, N_CHARS, CHARACTERS, MFCC)

    # encoder dimensionality
    encoder_timesteps = None
    encoder_features = MFCC

    # decoder dimensionality
    decoder_timesteps = None
    decoder_features = N_CHARS

    # construct model
    model = construct_model((encoder_timesteps, encoder_features), (decoder_timesteps, decoder_features))
    model.summary()

    # early stopping
    es_callback = EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=DELTA, restore_best_weights=True)

    # reduce learning rate on plateau
    rl_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=DELTA)

    # optimizer
    opt = Adam(learning_rate=0.001)

    # compile and fit model
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([x1, x2, x1_mask, x2_mask], y, epochs=EPOCHS, verbose=1, batch_size=64, shuffle=False, callbacks=[es_callback, rl_callback], validation_data=([x1_val, x2_val, x1_val_mask, x2_val_mask], y_val))

    # save model
    model.save("custom_encoder_decoder")

    return model

################## Section 2: Inference ##################

# reconstruct the model for inference
def reconstruct_encoder_decoder(model):
    # ENCODER
    
    # inputs
    encoder_inputs = Input(shape=(None, MFCC))
    encoder_mask_input = Input(shape=(None,), dtype=tf.bool)
    
    # lstm
    encoder_outputs, state_h_enc, state_c_enc = model.layers[3](encoder_inputs, mask=encoder_mask_input)
    
    # model
    encoder_model = Model([encoder_inputs, encoder_mask_input], [state_h_enc, state_c_enc])
    
    # DECODER
    
    # inputs 
    decoder_inputs = Input(shape=(1, N_CHARS))
    decoder_mask_input = Input(shape=(None,), dtype=tf.bool)
    
    # lstm
    decoder_state_input_h = Input(shape=(DECODER,))
    decoder_state_input_c = Input(shape=(DECODER,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h_dec, state_c_dec = model.layers[5](decoder_inputs, initial_state=decoder_states_inputs, mask=decoder_mask_input) 
    
    # output
    decoder_outputs = model.layers[6](decoder_outputs)
    
    # model
    decoder_model = Model([decoder_inputs, decoder_mask_input] + decoder_states_inputs, [decoder_outputs, state_h_dec, state_c_dec])
    
    return encoder_model, decoder_model


# decode one timestep at a time until "<E>" is reached
def decode_sequence(input_seq, input_mask, encoder_model, decoder_model):
    # pass the input through the encoder
    states_value = encoder_model.predict([input_seq, input_mask]) 

    # generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, N_CHARS))
    
    # populate the first character of target sequence with the start token
    target_seq[0, 0, CHARACTERS.index("<S>")] = 1

    # loop through decoder until stop condition activated
    stop_condition = False
    decoded_chars = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, np.zeros((1, target_seq.shape[1]))==0] + states_value)

        # sample a token
        sampled_char_index = np.argmax(output_tokens[0, -1, :])

        # get the characters from the index
        sampled_char = CHARACTERS[sampled_char_index]
        decoded_chars.append(sampled_char)

        # stop condition: either predict stop character or hit max length 
        if sampled_char == "<E>" or len(decoded_chars) > 15:
            stop_condition = True

        # update the target sequence for the next loop
        target_seq = np.zeros((1, 1, N_CHARS))
        target_seq[0, 0, sampled_char_index] = 1

        # update states
        states_value = [h, c]
    
    return decoded_chars[:-1]


# evaluate on a dataset
def evaluate(model, filepath, write_name, impaired):
    # encoder and decoder models
    encoder_model, decoder_model = reconstruct_encoder_decoder(model)
    
    # data extraction
    df = pandas.read_csv(filepath)
    X1, X2, Y, X1_mask, X2_mask = feature_extraction(filepath, NFFT, HOP, POWER, MELS, N_CHARS, CHARACTERS, MFCC)
    
    # predictions
    pred_data = []
    for i in range(len(df)):
        y_true = df["transcript"][i]
        y_pred = decode_sequence(np.reshape(X1[i,:,:], (1, X1.shape[1], X1.shape[2])), np.reshape(X1_mask[i,:], (1, X1_mask.shape[1])), encoder_model, decoder_model)
        y_pred = "".join(y_pred)
        
        # append results
        pred_data.append([y_true, y_pred])

    if impaired:
        impaired_errors(df, pred_data, write_name, SPEAKERS)
    else:
        standard_errors(df, pred_data, write_name)
    
    print("Inference complete for: %s" % write_name)


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="path to csv containing training data")
    # parser.add_argument("--train_dev_path", type=str, help="path to csv containing train-dev data")
    parser.add_argument("--dev_path", type=str, help="path to csv containing validation data")
    parser.add_argument("--test_path", type=str, help="path to csv containing test data")
    parser.add_argument("--train_dev_impaired", type=int, default=1, help="1 if just UASpeech (default), 0 if other data included")
    args = parser.parse_args()

    # training
    model = train_model(args.train_path, args.dev_path)
    
    # inference
    evaluate(model, args.train_path, "train", args.train_dev_impaired==1)
    # evaluate(model, args.train_dev_path, "train_dev", args.train_dev_impaired==1)
    evaluate(model, args.dev_path, "dev", True)
    evaluate(model, args.test_path, "test", True)


if __name__=="__main__":
    main()
        
