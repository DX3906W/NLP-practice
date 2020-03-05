import numpy as np
from itertools import chain

#generate all possible wrong spelling candidate
def generate_edit_one(word):

    candidate = []

    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(str[:i],str[i:]) for i in range (len(str) + 1)]

    inserts = [L + c + R for L ,R in splits for c in letters]           #[a-z]apple,
    deletes = [L + R[1:] for L,R in splits if R]
    replaces = [L + c + R[1:] for L,R in splits if R for c in letters ]

    candidate.append(list(set(inserts+deletes+replaces)))

def generate_condidate(word, threshold):

    candidates = {}
    candidates[0] = generate_edit_one(word)

    for i in range(1, threshold):
        candidates[i] = [generate_edit_one(w) for w in candidates[i]]

    return list(set(chain.from_iterable(candidates.keys())))

#load all data
def load_dataset():
    root_path = '../dataSet/SC/'
    vocab_path = root_path + 'vocab.txt'
    spell_error = root_path + 'spell-error.txt'
    testdata_path = root_path + 'textdata.txt'

    vocab = []
    for v in open(vocab_path, 'r'):
        vocab.append(v)



def channel_prob(sentence):

    channel_prob = {}

    for line in open('spell-errors.txt'):
        items = line.split(":")
        correct = items[0].strip()
        mistakes = [item.strip() for item in items[1].strip().split(",")]
        channel_prob[correct] = {}
        for mis in mistakes:
            channel_prob[correct][mis] = 1.0/len(mistakes)

    return channel_prob
#
def correct():
    load_dataset()
