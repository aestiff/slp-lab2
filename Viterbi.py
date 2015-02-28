'''
Created on Feb 26, 2015

@author: stiff
'''

import numpy as np
from numpy import Infinity
import prob1
import nltk

class Viterbi:
    def __init__(self, transitionProbs, emissionProbs):
        # convert dictionaries to numpy matrices
        self.tagIndex = set([otherTag for (tag, otherTag) in transitionProbs.keys()])
        self.tagIndex = sorted(self.tagIndex.union(set([tag for (tag,otherTag) in transitionProbs.keys()])))        
        self.wordIndex = sorted(set([word for (word, tag) in emissionProbs.keys()]))    
        self.transitionProbs = np.ndarray((len(self.tagIndex),len(self.tagIndex)))
        self.transitionProbs.fill(Infinity)
        for (tag, otherTag) in transitionProbs.keys():
            self.transitionProbs[self.tagIndex.index(tag),self.tagIndex.index(otherTag)]=transitionProbs[(tag,otherTag)]
        self.emissionProbs = np.ndarray((len(self.wordIndex),len(self.tagIndex)))
        self.emissionProbs.fill(Infinity)
        for (word, tag) in emissionProbs.keys():
            self.emissionProbs[self.wordIndex.index(word),self.tagIndex.index(tag)]=emissionProbs[(word,tag)]
    ''' 
    sentence is a list of word strings 
    '''
    def findShortestPath(self, sentence):
        viterbi = np.ndarray((len(sentence),len(self.tagIndex)))
        backpoint = np.ndarray((len(sentence),len(self.tagIndex)),dtype=int)
        viterbi.fill(Infinity)
        for i in range(len(self.tagIndex)):
            # initialize each state according to its neg log prob given the start state
            viterbi[0,i] = self.transitionProbs[i,self.tagIndex.index("start")]+ \
                self.emissionProbs[self.wordIndex.index(sentence[0]),i]
            backpoint[0,i] = self.tagIndex.index("start")
        for j in range(1,len(sentence)):
            for i in range(len(self.tagIndex)):
                # here we take the negative log probs from the previous states, add appropriate 
                # transition (neg log) probabilities to each one (they're ordered), 
                # find the minimal value among them, and add the appropriate emission
                # (neg log) prob for this state/word combo 
                transitionArray = viterbi[[j-1],:]+self.transitionProbs[[i],:]
                viterbi[j,i] = np.min(transitionArray) \
                    +self.emissionProbs[[self.wordIndex.index(sentence[j])],[i]]
                backpoint[j,i] = np.argmin(transitionArray)
        lastIdx = np.argmin(viterbi[[len(sentence)-1],:])
        result = [(sentence[len(sentence)-1],self.tagIndex[lastIdx])]
        for k in range(len(sentence)-1, 0,-1):
            result.insert(0, (sentence[k-1],self.tagIndex[backpoint[k,lastIdx]]))
            lastIdx = backpoint[k,lastIdx]
        return result
                    
            
        

if __name__ == '__main__':
#    tData = {("NN","NN"):2,("VB","NN"):1,("VB","VB"):2,("NN","VB"):3, ("NN","start"):6,("VB","start"):3}
#    eData = {("dog","NN"):5, ("bark","VB"):4, ("cat","NN"):5}
    full_training=nltk.corpus.treebank.tagged_sents()[0:3500]
    training_set1=full_training[0:1750]
    training_set2=full_training[1750:]
    test_set=nltk.corpus.treebank.tagged_sents()[3500:]

    print("counting...")
    (wrdtagcount_table,tagtagcount_table) =    prob1.calculateprobtables(full_training)
    (wrdtagcount_table,tagtagcount_table) =    prob1.calculateprobtables(training_set1)

    print("viterbing...")
    v = Viterbi(tagtagcount_table,wrdtagcount_table)
    for i in range(10):
        print(training_set1[i])
        print(v.findShortestPath([word for (word,tag) in training_set1[i]]))
