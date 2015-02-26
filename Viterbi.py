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
        viterbi = np.zeros((len(sentence),len(self.tagIndex)))
        for i in range(len(self.tagIndex)):
            viterbi[0,i] = self.transitionProbs[i,self.tagIndex.index("start")]+ \
                self.emissionProbs[self.wordIndex.index(sentence[0]),i]
        for j in range(1,len(sentence)):
            for i in range(len(self.tagIndex)):
                viterbi[j,i] = min(viterbi[j-1]+self.transitionProbs[i]+self.emissionProbs[self.wordIndex.index(sentence[j])])
        print(min(viterbi[len(viterbi)-1]))
            
        

if __name__ == '__main__':
    tData = {("NN","NN"):2,("VB","NN"):1,("VB","VB"):2,("NN","VB"):3, ("start","NN"):6,("start","VB"):3}
    eData = {("dog","NN"):5, ("bark","VB"):4}
    v = Viterbi(tData,eData)
    print(v.findShortestPath(["dog", "bark"]))