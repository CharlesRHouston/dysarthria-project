# find min edit distance between two strings

# modules
import numpy as np

# Levenshtein distance
def min_edit_distance(str1, str2):
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