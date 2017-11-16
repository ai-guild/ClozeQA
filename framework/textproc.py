'''
    Text Processing Utilities

'''
from nltk import word_tokenize
from info import *
import resources as R


'''
    Reduce text to set of words

'''
def reduce_(dataitems):
    words = []
    for item in dataitems:
        words.extend(word_tokenize(
            item['question'] + item['context']))

    return set(words)

'''
    Dump vocabulary to disk

'''
def dump_vocabulary(words):
    with open(R.VOCAB, 'w') as f:
        for w in words:
            f.write(w)
            f.write('\n')
    # print info
    Ip(':: {} words cached'.format(len(words)))
