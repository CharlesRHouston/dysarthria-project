# take in predictions text file and produce WERs grouped by word category

from levenshtein import min_edit_distance
import pandas
import argparse

# remove dash from word
def remove_dash(word):
	new_word = []
	for char in word:
		if char == '-':
			continue
		else:
			new_word.append(char)
	return ''.join(new_word)

# dictionary of with words as keys and categories as valeus
def createCategoryDict(file):
	wordToCategory = {}
	df = pandas.read_excel(file, sheet_name = "Word_filename")
	for i in range(len(df)):
		if df["FILE NAME"][i][0:2] in ("B1", "B3"):
			continue
		key = remove_dash(df["WORD"][i].lower())
		if df["FILE NAME"][i][0:2] == "B2":
			wordToCategory[key] = "UW"
		elif df["FILE NAME"][i][0:2] == "CW":
			wordToCategory[key] = "CW"
		else:	
			wordToCategory[key] = df["FILE NAME"][i][0]
	return wordToCategory


# calculate WER by category
def errors(new_df, categoryDict, categoryErrors, output_name):

	for element in new_df:
		# edits and length
		edits = min_edit_distance(element[1], element[2])
		length = len(element[1])

		# identify category
		category = categoryDict[element[1][0]]

		# category error
		categoryErrors[category][0] += edits
		categoryErrors[category][1] += length

		# total error
		categoryErrors["Total"][0] += edits
		categoryErrors["Total"][1] += length

	for key, values in categoryErrors.items():
		print(key)
		print(round(values[0]/values[1]*100, 2))
		print("\n")

	# write prediction to text file
	with open(output_name + ".txt", "a") as text_file:
		for key, values in categoryErrors.items():
			text_file.write(f"{key}: {round(values[0]/values[1]*100, 2)} %\n")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--speaker_wordlist', default="C:\\Users\\charl\\Desktop\\Thesis\\Data\\UASpeech\\speaker_wordlist.xls", help='Path to the spearker wordlist xls file')
	parser.add_argument('--predictions', default="C:\\Users\\charl\\Desktop\\Instance\\Inference\\results\\fine-tune-aug\\predictions.txt", help='Path to predictions txt file')
	parser.add_argument('--output_name', default="predictions", help='Name of output txt results file')
	args = parser.parse_args()

	# create dictionary
	categoryDict = createCategoryDict(args.speaker_wordlist)
	
	# category errors storage
	categoryErrors = {}
	for element in ("C", "D", "L", "CW", "UW", "Total"):
		categoryErrors[element] = [0, 0]

	# read in text file
	file = open(args.predictions)
	df = file.readlines()

	# line by line extract data
	count = 0
	new_df = []
	while count < len(df):
		speaker = df[count].replace("Speaker: ", "").replace("\n", "")
		actual = df[count+1].replace("Actual: ", "").replace("\n", "").split(" ")
		predicted = df[count+2].replace("Predicted: ", "").replace("\n", "").split(" ")
		new_df.append([speaker, actual, predicted])
		count += 4

	errors(new_df, categoryDict, categoryErrors, args.output_name)


if __name__ == "__main__":
	main()


