import pickle
import os

from datapoint import Datapoint

import resources as R


'''
    Dataset
      - ( structured_data )
      - ( list_of_dicts )

'''
class Dataset(object):

    def __init__(self, pipe=None, flush=False):

        if not flush and self.read(pipe.__name__):
            pass

        else:
            traindata, testdata = pipe()
            # create data points
            self.trainset = self.sort([ Datapoint(item) # sort list
                for item in traindata ])
            self.testset  = self.sort([ Datapoint(item) 
                    for item in testdata ])

            # save to disk
            self.write(pipe.__name__)

    '''
        Sort datapoints based on context length

    '''
    def sort(self, data):
        return sorted(data, 
                key = lambda x : x.context_len())

    '''
        Cache datapoints

    '''
    def write(self, pipe_id):
        with open(R.TRAIN.format(pipe_id), 'wb') as f:
            pickle.dump(self.trainset, f)
        with open(R.TEST.format(pipe_id),  'wb') as f:
            pickle.dump(self.testset, f)

    '''
        Read from cache if availabel

    '''
    def read(self, pipe_id):
        # check if cache dir exists
        if not os.path.exists(R.CACHE):
            os.makedirs(R.CACHE)

        try:
            with open(R.TRAIN.format(pipe_id), 'rb') as f:
                self.trainset = pickle.load(f)
            with open(R.TEST.format(pipe_id), 'rb') as f:
                self.testset = pickle.load(f)
            return True
        except:
            pass
