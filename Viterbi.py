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
            wordIdx = 0
            try:
                wordIdx = self.wordIndex.index(sentence[0])
            except ValueError:
                wordIdx = self.wordIndex.index('UNK')
            viterbi[0,i] = self.transitionProbs[i,self.tagIndex.index("start")]+ \
                self.emissionProbs[wordIdx,i]
            backpoint[0,i] = self.tagIndex.index("start")
        for j in range(1,len(sentence)):
            for i in range(len(self.tagIndex)):
                # here we take the negative log probs from the previous states, add appropriate 
                # transition (neg log) probabilities to each one (they're ordered), 
                # find the minimal value among them, and add the appropriate emission
                # (neg log) prob for this state/word combo 
                transitionArray = viterbi[[j-1],:]+self.transitionProbs[[i],:]
                wordIdx = 0
                try:
                    wordIdx = self.wordIndex.index(sentence[j])
                except ValueError:
                    wordIdx = self.wordIndex.index('UNK')
                viterbi[j,i] = np.min(transitionArray) + self.emissionProbs[[wordIdx],[i]]
                backpoint[j,i] = np.argmin(transitionArray)
        lastIdx = np.argmin(viterbi[[len(sentence)-1],:])
        result = [(sentence[len(sentence)-1],self.tagIndex[lastIdx])]
        for k in range(len(sentence)-1, 0,-1):
            result.insert(0, (sentence[k-1],self.tagIndex[backpoint[k,lastIdx]]))
            lastIdx = backpoint[k,lastIdx]
        return result
'''
This method assumes that both sets are ordered in the same fashion
and contain the same sentences.
'''                    
def countErrors(groundTruthSet, labelledSet):
    errorCount = 0
    totalCount = 0
    for i in range(len(groundTruthSet)):
        for j in range(len(groundTruthSet[i])):
            totalCount += 1
            (word1,tag1) = groundTruthSet[i][j]
            (word2,tag2) = labelledSet[i][j]
            if word1 == word2:
                if tag1 != tag2:
                    errorCount += 1
            else:
                raise Exception("Sentences don't match")
    return float(errorCount)/totalCount

if __name__ == '__main__':
#    tData = {("NN","NN"):2,("VB","NN"):1,("VB","VB"):2,("NN","VB"):3, ("NN","start"):6,("VB","start"):3}
#    eData = {("dog","NN"):5, ("bark","VB"):4, ("cat","NN"):5}
    full_training=nltk.corpus.treebank.tagged_sents()[0:3500]
    training_set1=full_training[0:1750]
    training_set2=full_training[1750:]
    test_set=nltk.corpus.treebank.tagged_sents()[3500:]

    print("counting...")
    (wrdtagcount_table,tagtagcount_table) =    prob1.calculateprobtables(full_training)
    #(wrdtagcount_table,tagtagcount_table) =    prob1.calculateprobtables(training_set1)

    print("viterbing...")
#    labelTest = []
    v = Viterbi(tagtagcount_table,wrdtagcount_table)
#   for i in range(len(full_training)):
#       labelTest.append(v.findShortestPath([word for (word,tag) in full_training[i]]))
#   print("Training error: " + str(countErrors(full_training, labelTest)))

    labelTest = []
    for i in range(len(test_set)):
        labelTest.append(v.findShortestPath([word for (word,tag) in test_set[i]]))
    print("Test error: " + str(countErrors(test_set, labelTest)))

'''
Step 5
'''
convergence = False

labelTest = []
testSetLabels = []
(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(training_set1)
v = Viterbi(tagtagcount_table,wrdtagcount_table)
for i in range(len(training_set2)):
        labelTest.append(v.findShortestPath([word for (word,tag) in training_set2[i]]))
labelTest.extend(training_set1)
(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(labelTest)
v = Viterbi(tagtagcount_table,wrdtagcount_table)
for i in range(len(test_set)):
    testSetLabels.append(v.findShortestPath([word for (word,tag) in test_set[i]]))
errorRate = countErrors(test_set, testSetLabels)
print('Test Error: '+str(errorRate))

while not convergence:
    newModel = []
    testSetLabels = []
    for i in range(len(training_set2)):
        newModel.append(v.findShortestPath([word for (word,tag) in training_set2[i]]))
    newModel.extend(training_set1)
    (wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(newModel)
    v = Viterbi(tagtagcount_table,wrdtagcount_table)
    for i in range(len(test_set)):
        testSetLabels.append(v.findShortestPath([word for (word,tag) in test_set[i]]))
    errorRate = countErrors(test_set, testSetLabels)
    print 'Test Error: '+str(errorRate)
    if errorRate <= 0.01:
	convergence = True 
