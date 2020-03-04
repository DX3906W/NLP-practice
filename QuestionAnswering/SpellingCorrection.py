
#generate all possible wrong spelling candidate
def generateCandidate(word, threshold):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replacement = [letters[j:j+i] for i in range(threshold) for j in range(len(letters))]

    

#load all data
def load_dataset():
    root_path = '../dataSet/SC/'
    vocab_path = root_path + 'vocab.txt'
    spell_error = root_path + 'spell-error.txt'
    testdata_path = root_path + 'textdata.txt'

    vocab = []
    for v in open(vocan_path, 'r'):
        vocab.append(v)

#
def correct(word):
