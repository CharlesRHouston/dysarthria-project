# modules
import argparse
import os
import pandas as pd
import numpy as np


# valid speakers
# SPEAKERS = ("F02", "F03", "F04", "F05", "M01", "M04", "M05", "M07", "M08", "M09", "M10", "M11", "M12", "M14", "M16")
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
    df = pd.read_excel(file, sheet_name="Word_filename")
    for i in range(len(df)):
        if '-' in df["WORD"][i]:
            keyToLabel[df["FILE NAME"][i]] = remove_dash(df["WORD"][i].lower())
        else:
            keyToLabel[df["FILE NAME"][i]] = df["WORD"][i].lower()
    return keyToLabel


# add row to data frame
def add_row(path, file, keyToLabel, word_key, df, op_sys):
    # file path
    if op_sys == 1:
        file_path = path + "/" + file
    elif op_sys == 2:
        file_path = path + "\\" + file
    # file size
    file_size = os.path.getsize(file_path)
    # target label
    file_label = keyToLabel[word_key]
    # create new row and append to data frame
    new_row = {"wav_filename": file_path, "wav_filesize": file_size, "transcript": file_label}
    df = df.append(new_row, ignore_index = True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Create train, validation and test set csv files for fine-tuning")
    parser.add_argument("--path_name", type = str, \
        help = "path to folder containing wav files for F02, F03, etc. e.g. /mnt/c/Users/charl/Desktop/UASpeech")
    parser.add_argument("--keyword_file", type = str, default = "speaker_wordlist.xls",\
        help = "path to xls file containing mapping from keys to words; default 'speaker_wordlist.xls'")
    parser.add_argument("--set", type = int, default = 1, \
        help = "1 if training and validation sets, 2 if test set")
    parser.add_argument("--os", type = int, default = 1, \
        help = "1 if UNIX, 2 if WINDOWS; default=UNIX")
    args = parser.parse_args()

    # create data frame
    df = pd.DataFrame({"wav_filename": [], "wav_filesize": [], "transcript": []})

    # create label dictionary
    keyToLabel = createLabelDict(args.keyword_file)

    # loop through files
    for path, dir, files in os.walk(args.path_name):
        speaker = os.path.basename(path)
        # ensure folder is speaker
        if speaker not in SPEAKERS:
            continue
        # loop through files
        for file in files:
            # ensure file is wav
            if "wav" not in file:
                continue
            # split and get batch and word key
            components = file.split("_")
            batch = components[1]
            word_key = components[2]
            
            # uncommon word key
            if word_key[0:2] == "UW":
                word_key = batch + "_" + word_key
            
            # logic
            if args.set == 1 and batch in ("B1", "B3"):
                df = add_row(path, file, keyToLabel, word_key, df, args.os)

            if args.set == 2 and batch == "B2":
                df = add_row(path, file, keyToLabel, word_key, df, args.os)
                
    # change byte size to integer
    df = df.astype({"wav_filesize": np.int32})
    # shuffle data
    df = df.sample(frac=1, random_state=2021)
    # write data frame to csv
    if args.set == 1:
        # split train and train-dev 85/15
        ninety = int(len(df)*0.85)
        df.iloc[0:ninety].to_csv("train.csv", index = False)
        df.iloc[ninety:].to_csv("dev.csv", index = False)
        print("completed train.csv and dev.csv")
    elif args.set == 2:
        df.to_csv("test.csv", index = False)
        print("completed test.csv")


if __name__ == '__main__':
    main()
