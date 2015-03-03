'''
Created on Mar 2, 2015

@author: stiff
'''
import nltk

class ForwardBackward(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        x = nltk.tag.hmm.HiddenMarkovModelTagger()