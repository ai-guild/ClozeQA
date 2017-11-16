'''
    Global collection of resources, pointer to resources

'''

# link to essential folders and files
CACHE    = '.cache/'
DATA     = '../datasets/'
CBT      = '../datasets/CBTest/data/'
LOOKUP   = 'lookup/'
VOCAB    = 'lookup/vocabulary.txt'
EMB      = '.cache/embeddings.{}.bin' # dimensions
GLOVE    = '../datasets/glove/glove.6B.{}d.txt' # dimensions
PARAMS   = '.cache/params.{}.pkl' # model name/timestamp
TRAIN    = '.cache/train.{}.pkl'  # pipe id
TEST     = '.cache/test.{}.pkl'   # pipe id

# Glove dimensions
DIMS     = [ 50, 100, 200, 300 ]
