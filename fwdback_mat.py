import math
import prob1
import nltk
import numpy as np

#Computing forward probabilities
def calfwdprobs(transitionProbs,emissionProbs,sentence,wordIndex,taglist):
    alpha = np.ndarray((len(taglist),(len(sentence)+2)))
    alpha.fill(math.log(0.001))
    #First Step (from start to other tags)
    for i in range(len(taglist)):
        # initialize each state according to its neg log prob given the start state
        wordIdx = 0
        try:
            wordIdx = wordIndex.index(sentence[0])
        except ValueError:
            wordIdx = wordIndex.index('UNK')
        alpha[i,1] = transitionProbs[i,tagIndex.index("start")]+ emissionProbs[wordIdx,i]
        #################################################        
    
    #Next steps    
    lst = []
    for time in range(2,len(sentence)+1):
        for i in range(len(taglist)):
            lst = []
            for tim1 in range(len(taglist)):
                wordIdx = 0
                try:
                    wordIdx = wordIndex.index(sentence[time-1])
                except ValueError:
                    wordIdx = wordIndex.index('UNK')
                lst.append(alpha[tim1,time-1] + transitionProbs[i,tim1]+ emissionProbs[wordIdx,i])
            #print(lst)
            alpha[i,time] = logExpSumTrick(lst)
    #####################################################
    
    #Last step ( when you are at stop )        
    lst = []
    for i in range(len(taglist)):
        lst.append(alpha[i,len(sentence)] + transitionProbs[taglist.index("stop"),i])
                
    alpha[(tagIndex.index("stop")),(len(sentence)+1)] = logExpSumTrick(lst)
    ######################################################
    
    return alpha

#Computing backward probabilities
def calbackprobs(transitionProbs,emissionProb,sentence,wordIndex,taglist):

    
    beta = np.ndarray((len(taglist),(len(sentence)+2)))
    beta.fill(math.log(0.001))
    
    
    #First Step (from stop to other tags)
    for i in range(len(taglist)):
        beta[i,len(sentence)] = transitionProbs[taglist.index('stop'),i]
    
    #######################################        
    
    #Next steps
    lst = []    
    for time in reversed(range(1,len(sentence))):
        for i in range(len(taglist)):
            lst = []
            for tip1 in range(len(taglist)):
                wordIdx = 0
                try:
                    wordIdx = wordIndex.index(sentence[time])
                except ValueError:
                    wordIdx = wordIndex.index('UNK')
                lst.append(beta[tip1,time+1] + transitionProbs[tip1,i]+ emissionProbs[wordIdx,tip1])
            beta[i,time] = logExpSumTrick(lst)
    #########################################        
            
    #Last step ( when you are at start )        
    lst = []
    for i in range(len(taglist)):
        wordIdx = 0
        try:
            wordIdx = wordIndex.index(sentence[0])
        except ValueError:
            wordIdx = wordIndex.index('UNK')
        lst.append(beta[i,1] + transitionProbs[i,taglist.index('start')]+ emissionProbs[wordIdx,i])        
    beta[taglist.index("start"),0] = logExpSumTrick(lst)
    
    ##########################################
    
    
    return beta
    
    
def logExpSumTrick(array):
    largest = max(array)
    total = 0.0
    for prob in array:
        total = total + math.exp(prob-largest)
    total = math.log(total)
    total = largest + total
    
    return total        
    
full_training=nltk.corpus.treebank.tagged_sents()[0:100]
#training_set1=full_training[0:1750]
#training_set2=full_training[1750:]
#test_set=nltk.corpus.treebank.tagged_sents()[3500:]



print("counting...")
(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(full_training)
#(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(training_set1)

#Create matrices
tagIndex = set([otherTag for (tag, otherTag) in tagtagcount_table.keys()])
tagIndex = sorted(tagIndex.union(set([tag for (tag,otherTag) in tagtagcount_table.keys()])))        
wordIndex = sorted(set([word for (word, tag) in wrdtagcount_table.keys()]))
    
transitionProbs = np.ndarray((len(tagIndex),len(tagIndex)))
transitionProbs.fill(math.log(0.001))

for (tag, otherTag) in tagtagcount_table.keys():
    transitionProbs[tagIndex.index(tag),tagIndex.index(otherTag)]=tagtagcount_table[(tag,otherTag)]

emissionProbs = np.ndarray((len(wordIndex),len(tagIndex)))
emissionProbs.fill(math.log(0.001))
for (word, tag) in wrdtagcount_table.keys():
    emissionProbs[wordIndex.index(word),tagIndex.index(tag)]=wrdtagcount_table[(word,tag)]



print("Forward-Backward");
num=0
for sentence in full_training:
    num = num + 1
    print(num)
    Fprobs = calfwdprobs(transitionProbs,emissionProbs,sentence,wordIndex,tagIndex)
    #print Fprobs
    Bprobs = calbackprobs(transitionProbs,emissionProbs,sentence,wordIndex,tagIndex)
    #print Bprobs

