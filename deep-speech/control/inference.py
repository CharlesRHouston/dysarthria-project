#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json
import os
import pandas 

from deepspeech import Model, version
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# control speakers
SPEAKERS = ("CF02", "CF03", "CF04", "CF05", "CM01", "CM04", "CM05", "CM06", "CM08", "CM09", "CM10", "CM12", "CM13")

def remove_dash(word):
    new_word = []
    for char in word:
        if char == '-':
            continue
        else:
            new_word.append(char)
    return ''.join(new_word)

def createLabelDict(file):
    keyToLabel = {}
    df = pandas.read_excel(file, sheet_name = "Word_filename")
    for i in range(len(df)):
        if '-' in df["WORD"][i]:
            keyToLabel[df["FILE NAME"][i]] = remove_dash(df["WORD"][i].lower())
        else:
            keyToLabel[df["FILE NAME"][i]] = df["WORD"][i].lower()
    return keyToLabel

def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json.dumps(json_result, indent=2)



class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        print('DeepSpeech ', version())
        exit(0)


def minEditDistance(str1, str2):
    # initialise dynamic programming array
    editMatrix = np.zeros((len(str1) + 1, len(str2) + 1))
    
    # populate first column
    for i in range(len(str1) + 1):
        editMatrix[i, 0] = i
    
    # populate first row
    for i in range(len(str2) + 1):
        editMatrix[0, i] = i
    
    # populate remainder of editMatrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                editMatrix[i, j] = editMatrix[i - 1, j - 1]
            else:
                substitution = editMatrix[i - 1, j - 1] + 1
                insertion = editMatrix[i, j - 1] + 1
                deletion = editMatrix[i - 1, j] + 1
                editMatrix[i, j] = min(substitution, insertion, deletion)

    return int(editMatrix[-1, -1])


def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=False,
                        help='Path to the external scorer file')
    parser.add_argument('--audio', required=True,
                        help='Path to the csv file to run (WAV format)')
    parser.add_argument('--keys', required=True,
                        help='Path to the csv file of target keys')
    parser.add_argument('--beam_width', type=int,
                        help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float,
                        help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float,
                        help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--version', action=VersionAction,
                        help='Print version and exits')
    parser.add_argument('--extended', required=False, action='store_true',
                        help='Output string from extended metadata')
    parser.add_argument('--json', required=False, action='store_true',
                        help='Output json from metadata with timestamp of each word')
    parser.add_argument('--candidate_transcripts', type=int, default=3,
                        help='Number of candidate transcripts to include in JSON output')
    parser.add_argument('--hot_words', type=str,
                        help='Hot-words and their boosts.')
    parser.add_argument('--results_name', type=str, default = "results",
                        help='name of results csv')
    parser.add_argument('--predictions_name', type=str, default = "predictions",
                        help='name of predictions txt')
    args = parser.parse_args()

    print('Loading model from file {}'.format(args.model), file=sys.stderr)
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(args.model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if args.beam_width:
        ds.setBeamWidth(args.beam_width)

    desired_sample_rate = ds.sampleRate()

    if args.scorer:
        print('Loading scorer from files {}'.format(args.scorer), file=sys.stderr)
        scorer_load_start = timer()
        ds.enableExternalScorer(args.scorer)
        scorer_load_end = timer() - scorer_load_start
        print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

        if args.lm_alpha and args.lm_beta:
            ds.setScorerAlphaBeta(args.lm_alpha, args.lm_beta)

    if args.hot_words:
        print('Adding hot-words', file=sys.stderr)
        for word_boost in args.hot_words.split(','):
            word,boost = word_boost.split(':')
            ds.addHotWord(word,float(boost))

    # create dictionary to convert keys to labels 
    keyToLabel = createLabelDict(args.keys)

    # initialise results: very low, low, mid, high, total
    wer_total = 0
    wer_incorrect = 0  
    cer_total = 0
    cer_incorrect = 0

    # initialise speaker WER results: speaker 1 to 15
    wer_speaker_total = [0]*len(SPEAKERS)
    wer_speaker_incorrect = [0]*len(SPEAKERS)
    
    # set count to zero
    count = 0

    # iterate through files
    df = pandas.read_csv(args.audio)
    for clip_path in df["wav_filename"]:
        # read in audio
        fin = wave.open(clip_path, 'rb')
        fs_orig = fin.getframerate()
        if fs_orig != desired_sample_rate:
            print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
            fs_new, audio = convert_samplerate(clip_path, desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        audio_length = fin.getnframes() * (1/fs_orig)
        fin.close()

        # inference
        if args.extended:
            y_pred = metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0])
        elif args.json:
            y_pred = metadata_json_output(ds.sttWithMetadata(audio, args.candidate_transcripts))
        else:
            y_pred = ds.stt(audio)

        # obtain information from clip name
        clip = clip_path.split("/")[-1]
        components = clip.split("_")
        speaker = components[0]
        batch = components[1]
        word_key = components[2]

        # find label
        if word_key[0:2] == "UW":
            word_key = batch + "_" + word_key
        y_true = keyToLabel[word_key]

        # speaker index
        idx_speaker = list(SPEAKERS).index(speaker)

        # WORD ERROR RATE
        
        # edits and length
        edits = minEditDistance(y_pred.split(" "), y_true.split(" "))
        length = len(y_true.split(" "))
        
        # total
        wer_incorrect += edits
        wer_total += length

        # speaker
        wer_speaker_incorrect[idx_speaker] += edits
        wer_speaker_total[idx_speaker] += length

        # CHARACTER ERROR RATE
        
        # edits
        edits = minEditDistance(y_pred, y_true)
        
        # total
        cer_incorrect += edits
        cer_total += len(y_true)

        # write prediction to text file
        with open(args.predictions_name + ".txt", "a") as text_file:
            text_file.write(f"Speaker: {speaker}\n")
            text_file.write(f"Actual: {y_true}\n")
            text_file.write(f"Predicted: {y_pred}\n\n")

        # counter
        count += 1
        if count % 50 == 0:
            print(f"Examples inferred: {count}")
    
    # print out error results
    print(f"Total word error rate: {round(wer_incorrect/wer_total*100, 2)} %")
    print(f"Total character error rate: {round(cer_incorrect/cer_total*100, 2)} %")
    
    # intelligibility results
    wer_row = [round(wer_incorrect/wer_total*100, 2)]
    cer_row = [round(cer_incorrect/cer_total*100, 2)]
    
    # create data frame for intelligibility results
    df = pandas.DataFrame([wer_row, cer_row], columns = ["Total"], index = ["WER", "CER"])
    
    # write results to a csv file
    name = args.results_name + ".csv"
    df.to_csv(name, index = True)

    # speaker results
    speaker_row = []
    for i in range(len(wer_speaker_incorrect)):
        speaker_row.append(round(wer_speaker_incorrect[i]/wer_speaker_total[i]*100, 2))
    
    # create data frame for speaker results
    df = pandas.DataFrame([speaker_row], columns = list(SPEAKERS))
    
    # write results to a csv file
    name = "speakers.csv"
    df.to_csv(name, index = False)
    

if __name__ == '__main__':
    main()