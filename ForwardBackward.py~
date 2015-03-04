'''
Created on Mar 2, 2015

@author: stiff
'''
import nltk
import numpy as np
import math

class ForwardBackward(object):
    '''
    classdocs
    '''


    def __init__(self, vocab, states):
        '''
        Vocab is a list of words.
        States is a list of tags.
        '''
        #TODO: initialize the vocab and states lists for class-level use
        
    def train(self, observations):
        '''
        Observations is a list of sentences, where each sentence is a list of untagged words.
        '''
        # initialize transmission, emission probs
        A = np.ndarray() #TODO: initialize this correctly
        B = np.ndarray() #TODO: initialize this correctly
        
        converged = False
        while not converged:
            oldA = A
            oldB = B
            for sent in observations:
                alpha = self._forward(sent)
                beta = self._backward(sent)
                #TODO: figure out gamma and ksi (E-step)
                #TODO: recalculate transition, emissions (A and B; M-step)
            if oldA - A < 0.01 and oldB - B < 0.01: #TODO: fix this. 
                #so far it's dumb, not sure how to measure convergence yet
                converged = True 
    
        
        
    def _forward(self, sentence):
        #TODO: adjust this so that it's doing the forward algorithm,
        # instead of Viterbi
        alpha = np.ndarray((len(sentence)+2,len(self.tagIndex)))
        alpha.fill(Infinity)
        for i in range(len(self.tagIndex)):
            # initialize each state according to its neg log prob given the start state
            wordIdx = 0
            try:
                wordIdx = self.wordIndex.index(sentence[0])
            except ValueError:
                wordIdx = self.wordIndex.index('UNK')
            alpha[0,i] = self.transitionProbs[i,self.tagIndex.index("start")]+ \
                self.emissionProbs[wordIdx,i]
        for j in range(1,len(sentence)):
            for i in range(len(self.tagIndex)):
                # here we take the negative log probs from the previous states, add appropriate 
                # transition (neg log) probabilities to each one (they're ordered), 
                # find the minimal value among them, and add the appropriate emission
                # (neg log) prob for this state/word combo 
                transitionArray = alpha[[j-1],:]+self.transitionProbs[[i],:]
                wordIdx = 0
                try:
                    wordIdx = self.wordIndex.index(sentence[j])
                except ValueError:
                    wordIdx = self.wordIndex.index('UNK')
                alpha[j,i] = np.min(transitionArray) + self.emissionProbs[[wordIdx],[i]]
        return alpha

    def _backward(self, sentence):
        #TODO: implement backward algorithm
        beta = np.ndarray()
        return beta


def logExpSumTrick(lst):
    largest = max(lst)
    sum = 0
    for prob in lst:
	sum += math.exp(prob-largest)
    sum = math.log(sum)
    sum = largest + sum
	
    return sum
