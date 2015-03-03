'''
Created on Mar 2, 2015

@author: stiff
'''
import nltk
import numpy as np

class ForwardBackward(object):
    '''
    classdocs
    '''


    def __init__(self, vocab, states):
        '''
        Vocab is a list of words.
        States is a list of tags.
        '''
        #initialize the vocab and states lists for class-level use
        
    def train(self, observations):
        '''
        Observations is a list of sentences, where each sentence is a list of untagged words.
        '''
        # initialize transmission, emission probs
        A = np.ndarray()
        B = np.ndarray()
        
        converged = False
        while not converged:
            oldA = A
            oldB = B
            for sent in observations:
                alpha = self._forward(sent)
                beta = self._backward(sent)
                #figure out gamma and ksi (E-step)
                #recalculate transition, emissions (A and B; M-step)
            if oldA - A < 0.01 and oldB - B < 0.01: # this is dumb, not sure how to measure convergence yet
                converged = True 
    
        
        
    def _forward(self, sentence):
        alpha = np.ndarray((len(sentence),len(self.tagIndex)))
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