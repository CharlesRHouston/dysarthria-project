# custom module
from levenshtein import min_edit_distance

# impaired error
def impaired_errors(df, pred_data, write_name, SPEAKERS):
    # speaker errors
    speaker_errors = {}
    for speaker in SPEAKERS.keys():
        speaker_errors[speaker] = {"wer": [0, 0], "cer": [0, 0]}

    # intelligibility errors
    intell_errors = {}
    for level in ["Very Low", "Low", "Mid", "High", "Total"]:
        intell_errors[level] = {"wer": [0, 0], "cer": [0, 0]}

    # loop through predictions
    for i, entry in enumerate(pred_data):
        # predictions
        y_true = entry[0]
        y_pred = entry[1]

        # speaker
        speaker = df["wav_filename"][i].split("/")[-2]
        intelligibility = SPEAKERS[speaker]
        
        # word error rate
        edits = min_edit_distance(y_pred.split(" "), y_true.split(" "))
        total = len(y_true.split(" "))

        speaker_errors[speaker]["wer"][0] += edits
        intell_errors[intelligibility]["wer"][0] += edits
        intell_errors["Total"]["wer"][0] += edits

        speaker_errors[speaker]["wer"][1] += total
        intell_errors[intelligibility]["wer"][1] += total
        intell_errors["Total"]["wer"][1] += total

        # character error rate
        edits = min_edit_distance(y_true, y_pred)
        total = len(y_true)

        speaker_errors[speaker]["cer"][0] += edits
        intell_errors[intelligibility]["cer"][0] += edits
        intell_errors["Total"]["cer"][0] += edits

        speaker_errors[speaker]["cer"][1] += total
        intell_errors[intelligibility]["cer"][1] += total
        intell_errors["Total"]["cer"][1] += total

        # write prediction to text file
        with open(write_name + "_predictions.txt", "a") as text_file:
            text_file.write(f"Speaker: {speaker}\n")
            text_file.write(f"Actual: {y_true}\n")
            text_file.write(f"Predicted: {y_pred}\n\n")

    # write errors to text file
    with open(write_name + "_errors.txt", "w") as text_file:
        # write intelligibility errors
        for level in ["Very Low", "Low", "Mid", "High", "Total"]:
            text_file.write(f"WER [{level}]: {round(intell_errors[level]['wer'][0]/intell_errors[level]['wer'][1]*100, 2)} \n" )
            text_file.write(f"CER [{level}]: {round(intell_errors[level]['cer'][0]/intell_errors[level]['cer'][1]*100, 2)} \n\n")
        text_file.write("-----------------------------\n")

        # write speaker errors
        for speaker in SPEAKERS.keys():
            text_file.write(f"WER [{speaker}]: {round(speaker_errors[speaker]['wer'][0]/speaker_errors[speaker]['wer'][1]*100, 2)} \n" )
            text_file.write(f"CER [{speaker}]: {round(speaker_errors[speaker]['cer'][0]/speaker_errors[speaker]['cer'][1]*100, 2)} \n\n")

    print(f"WER: {round(intell_errors['Total']['wer'][0]/intell_errors['Total']['wer'][1]*100, 2)}")



# standard error
def standard_errors(df, pred_data, write_name):
    # storage
    wer = [0, 0]
    cer = [0, 0]

    # loop through predictions
    for entry in pred_data:
        # predictions
        y_true = entry[0]
        y_pred = entry[1]

        # wer
        wer[0] += min_edit_distance(y_pred.split(" "), y_true.split(" "))
        wer[1] += len(y_true.split(" "))

        # cer
        cer[0] += min_edit_distance(y_true, y_pred)
        cer[1] += len(y_true)

        # write prediction to text file
        with open(write_name + "_predictions.txt", "a") as text_file:
            text_file.write("Actual: %s \n" % y_true)
            text_file.write("Predicted: %s \n\n" % y_pred)

    wer = round(wer[0]/wer[1]*100, 2)
    cer = round(cer[0]/cer[1]*100, 2)

    # write errors to text file
    with open(write_name + "_errors.txt", "w") as text_file:
        text_file.write("WER: %s \n" % wer)
        text_file.write("CER: %s \n" % cer)
    
    print(f"WER: {wer}") 
