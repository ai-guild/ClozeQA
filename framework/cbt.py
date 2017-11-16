'''
    Children's Book Test
     Preprocessing (Text) Pipeline

'''
from nltk import word_tokenize
from tqdm import tqdm
from random import choice

from info import *
from textproc import *

import resources as R

DATAFILE = R.CBT + 'cbtest_{}_{}.txt'
TEST_FORMAT = 'test_2500ex'
TRAIN_FORMAT = 'train'
DTYPES = { 
        'NE', # Named Entity
        'CN', # Common Noun
        'P',  # Preposition
        'V'   # Verb
        }


'''
    Read CBT file from disk

'''
def read_file(filename):
    with open(filename) as f:
        raw_data = f.read()
        samples = raw_data.split('\n\n')[:-1]

    return samples

'''
    Extract content (context, question, candidates, answer)

'''
def extract_content_from_sample(sample):
    # read first 20 lines -> context
    context = [ ' '.join(l.split(' ')[1:]) 
            for l in sample.split('\n')[:20] ]
    assert len(context) == 20

    # combine with spaces
    context = ' '.join(context)

    # read 21st line -> query, answer, candidates
    query, answer, candidates = [ item 
            for item in sample.split('\n')[20].split('\t')
            if item ]

    # read query
    query = ' '.join(query.split(' ')[1:])

    # fetch candidates
    #  make sure num of candidates == 10
    candidates = sorted(candidates.split('|'), 
            key=lambda x : len(x), reverse=True)

    if len(candidates) < 10: # pad with empty strings
        candidates.extend([''] * (10 - len(candidates)))

    # truncate list
    candidates = candidates[:10]

    # mutate multi-words in candidates
    for i,c in enumerate(candidates):
        if len(word_tokenize(c)) > 1:
            tokens = word_tokenize(c)
            candidates[i] = max(tokens, key=lambda x : len(x))

    # TESTS
    assert len(candidates) == 10, candidates
    #assert len([ c for c in candidates 
    #    if c ]) == 10, candidates
    assert len(word_tokenize(answer)) == 1, answer
    # answer should be in candidates list
    assert answer in candidates, (answer, '\n', candidates)
    # check for query format
    assert 'XXXXX' in query, query
    assert len( [ w for w in candidates 
        if len(word_tokenize(w)) > 1 ] ) == 0, candidates

    return {
            'answer'           : answer,
            'context'          : context, 
            'question'         : query,
            'candidates'       : candidates,
            'answer_idx'       : candidates.index(answer),
            'candidate_labels' : fetch_candidate_labels(
                context, candidates)
            }

def extract_content(samples):
    Ip('[CBTest] Extracting content from samples')
    return [ extract_content_from_sample(sample) 
            for sample in tqdm(samples) ]

'''
    Index samples

'''
def index_samples(samples, offset=0):
    isamples = []
    for i,sample in enumerate(samples):
        sample['idx'] = offset + i
        isamples.append(sample)

    return isamples, offset + len(samples)

'''
    Index candidates

'''
def fetch_candidate_labels(context, candidates):
    return [ candidates.index(w) if w in candidates else 0
            for w in word_tokenize(context) ] # words in context

'''
    Pipe #1

      { 'context', 'question', 'answer', 'candidates' }

'''
def pipe1(dtype='NE', verbose=False, pause=False):

    def process(filename):
        # read from file
        #  extract content from samples
        return extract_content(read_file(filename))

        # print samples
        if pause:
            while True:
                highlight_answer(choice(samples))
                input()
                        
    # process train and test files separately
    #  create unique indices for samples
    train, offset = index_samples(process(
        DATAFILE.format(dtype, TRAIN_FORMAT)))
    test, _ = index_samples(process(
        DATAFILE.format(dtype, TEST_FORMAT)), offset)

    # dump combined vocabulary
    dump_vocabulary(reduce_(train + test))

    return train, test


if __name__ == '__main__':
    pipe1(pause=True)
