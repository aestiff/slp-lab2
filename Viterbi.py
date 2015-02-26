'''
Created on Feb 26, 2015

@author: stiff
'''

import numpy as np

class Viterbi:
    def __init__(self, transitionProbs, emissionProbs):
        self.tagIndex = sorted(set([tag for (tag, otherTag) in transitionProbs.keys()]))
        self.wordIndex = sorted(set([word for (word, tag) in emissionProbs.keys()]))    
        self.transitionProbs = np.ndarray((len(self.tagIndex),len(self.tagIndex)))
        for (tag, otherTag) in transitionProbs.keys():
            self.transitionProbs[self.tagIndex.index(tag),self.tagIndex.index(otherTag)]=transitionProbs[(tag,otherTag)]
        self.emissionProbs = np.zeros((len(self.wordIndex),len(self.tagIndex)))
        for (word, tag) in emissionProbs.keys():
            self.emissionProbs[self.wordIndex.index(word),self.tagIndex.index(tag)]=emissionProbs[(word,tag)]

    ''' 
    sentence is a list of word strings 
    '''
    def findShortestPath(self, sentence):
        viterbi = np.zeros(len(self.tagIndex),len(sentence))
        

if __name__ == '__main__':
    tData = {("NN","NN"):2,("VB","NN"):1,("VB","VB"):2,("NN","VB"):3}
    eData = {("dog","NN"):5, ("bark","VB"):4}
    v = Viterbi(tData,eData)
    print(v.emissionProbs)