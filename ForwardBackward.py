'''
Created on Mar 2, 2015

@author: stiff
'''
import numpy as np
import math
from numpy import newaxis

class ForwardBackward(object):
    '''
    classdocs
    '''


    def __init__(self, vocab, states, A, B):
        '''
        Vocab is a list of words.
        States is a list of tags.
        '''
        # initialize the vocab and states lists for class-level use
        self.vocab = vocab
        self.states = states
        self.A = A
        self.B = B
        
    def train(self, observations):
        '''
        Observations is a list of sentences, where each sentence is a list of untagged words.
        '''
        
        converged = False
        ii = 0
        while not converged:
            oldA = np.copy(self.A)
            oldB = np.copy(self.B)
            for sent in observations:
                sent = ['start'] + sent + ['stop']
                alpha = self._forward(sent)
                beta = self._backward(sent)
                # figure out gamma and ksi (E-step)
                gamma, ksi = self._eStep(alpha, beta, sent)
                #recalculate transition, emissions (A and B; M-step)
                self.A, self.B = self._mStep(gamma, ksi, self.A, self.B, sent)
                print("A:\n" + str(self.A))
                print("old A: \n" + str(oldA))
                print("B:\n" + str(self.B))
                print("Old B:\n" + str(oldB))   
            ii += 1
            if ii == 2: # np.max(oldA - A)< 0.01 and np.max(oldB - B)< 0.01:
                #so far it's dumb, not sure how to measure convergence yet
                converged = True 
    
        
        
    def _forward(self, sentence):
        alpha = np.ndarray((len(self.states),len(sentence)))
        alpha.fill(math.log(0.00001))
        tagIdx = self.states.index('start')
        alpha[tagIdx,0] = 0        
        for j in range(1,len(sentence)):
            for i in range(len(self.states)):
                # here we take the log probs from the previous states, add appropriate 
                # transition (log) probabilities to each one (they're ordered), 
                # sum them with log exp sum trick, and add the appropriate emission
                # (log) prob for this state/word combo 
                transitionArray = alpha[:,[j-1]]+self.A[[i],:]
                wordIdx = 0
                try:
                    wordIdx = self.vocab.index(sentence[j])
                except ValueError:
                    wordIdx = self.vocab.index('UNK')
                alpha[i,j] = logExpSumTrick(transitionArray) + self.B[[wordIdx],[i]]
        return alpha

    def _backward(self, sentence):
        beta = np.ndarray((len(self.states),len(sentence)))
        beta.fill(math.log(0.00001))
        tagIdx = self.states.index('stop')
        beta[tagIdx,len(sentence)-1] = 0        
        for j in range(len(sentence)-2,-1,-1):
            for i in range(len(self.states)):
                # here we take the log probs from the previous states, add appropriate 
                # transition (log) probabilities to each one (they're ordered), 
                # sum them with log exp sum trick, and add the appropriate emission
                # (log) prob for this state/word combo 
                wordIdx = 0
                try:
                    wordIdx = self.vocab.index(sentence[j+1])
                except ValueError:
                    wordIdx = self.vocab.index('UNK')
                #print("Beta:" + str(beta[:,[j+1]]))
                #print("A: " + str(self.A[:,[i]]))
                #print("B: " + str(self.B[[wordIdx],:].T))
                transitionArray = beta[:,[j+1]]+self.A[:,[i]]+self.B[[wordIdx],:].T
                #print("Trans: " + str(transitionArray))
                beta[i,j] = logExpSumTrick(transitionArray )
        return beta
    
    def _eStep(self, alpha, beta, sent):
        gamma = alpha+beta-alpha[self.states.index("stop"),alpha.shape[1]-1]
        print("Alpha:\n" + str(alpha))
        print("Beta:\n" + str(beta))
        print("Gamma:\n" + str(gamma))
        ksi = np.ndarray((len(sent)-1,len(self.states), len(self.states)))
        for t in range(len(sent)-1):
            ksi[t] = alpha[:,t] + self.A  + self.B[[self.vocab.index(sent[t+1])],:].T + beta[:,[t+1]] - alpha[self.states.index("stop"),alpha.shape[1]-1]
        print("Ksi:\n" + str(ksi))
        return gamma, ksi

    def _mStep(self, gamma, ksi, A, B, sent):
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                A[i,j] = logExpSumTrick(ksi[:,i,j])
        temp = np.ndarray((len(sent),len(self.states)))
        for t in range(len(sent)-1):
            for j in range(len(self.states)):
                temp[t,j] = logExpSumTrick(ksi[t,:,j])
        denom = np.ndarray(len(self.states))
        for j in range(len(self.states)):
            denom[j] = logExpSumTrick(temp[:,j])
        A = A / denom
        bDenom = logExpSumTrick(gamma,1)
        sentSet = sorted(set(sent))
        for word in sentSet:
            B[vocab.index(word),:] = (logExpSumTrick(np.take(gamma, [i for i,x in enumerate(sent) if x == word], 1),1)).T/bDenom.T
        return A, B 
    
def logExpSumTrick(array,axis=None):
    largest = np.max(array,axis,keepdims=True)
    #print("largest: " + str(largest))
    total = np.sum(np.exp(array-largest),axis,keepdims=True)
#     for prob in array:
#         total += math.exp(prob-largest)
    total = np.log(total)
    total = largest + total
    return total

if __name__ == '__main__':
    tData = {("NN","NN"):-2,("VB","NN"):-1,("VB","VB"):-2,("NN","VB"):-3, ("NN","start"):-6,("VB","start"):-3,("stop","VB"):-2,("stop","NN"):-4}
    eData = {("dog","NN"):-5, ("bark","VB"):-4, ("cat","NN"):-5, ("stop","stop"):0, ("start","start"):0}
    vocab = sorted(set([word for (word, tag) in eData.keys()]))
    states = set([otherTag for (tag, otherTag) in tData.keys()])
    states = sorted(states.union(set([tag for (tag,otherTag) in tData.keys()])))
    A = np.ndarray((len(states),len(states)))
    A.fill(math.log(0.00001))
    B = np.ndarray((len(vocab),len(states)))
    for (tag, otherTag) in tData.keys():
        A[states.index(tag),states.index(otherTag)]=tData[(tag,otherTag)]
    B.fill(math.log(0.001))
    for (word, tag) in eData.keys():
        B[vocab.index(word),states.index(tag)]=eData[(word,tag)]
    fb = ForwardBackward(vocab, states, A, B)
    fb.train([["dog","bark"]])
