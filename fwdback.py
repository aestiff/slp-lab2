import math
import prob1
import nltk
from nltk.corpus import treebank

#Computing forward probabilities
def calfwdprobs(A,B,sentence,taglist):

	fwdprobs = {}
	for tag in taglist:
		for i in range(len(sentence) + 2):
			fwdprobs[(tag,i)] = 0

	
	
	#First Step (from start to other tags)
	for tag in taglist:
		try:
			em_prob = B[(sentence[0][0],tag)] 
		except KeyError:
			em_prob = math.log(0.01)#B[('UNK',tag)]
		try:
			trans_prob = A[(tag,'start')] 
		except KeyError:
			trans_prob = math.log(0.01)
		fwdprobs[(tag,1)] = trans_prob + em_prob 
		
		
	#Next steps	
	lst = []
	for time in range(2,len(sentence)+1):
		for tag in taglist:
			lst = []
			for tagim1 in taglist:
				try:
					em_prob = B[(sentence[time-1][0],tag)] 
				except KeyError:
					em_prob = math.log(0.01)#B[('UNK',tag)]
				try:
					trans_prob = A[(tag,tagim1)] 
				except KeyError:
					trans_prob = math.log(0.01)
				lst.append(fwdprobs[(tagim1,time-1)] + trans_prob + em_prob)
			fwdprobs[(tag,time)] = logExpSumTrick(lst)
			
	#Last step ( when you are at stop )		
	lst = []
	for tag in taglist:
		try:
			trans_prob = A[('stop',tag)] 
		except KeyError:
			trans_prob = math.log(0.01)
		lst.append(fwdprobs[(tag,len(sentence))] + trans_prob)		
	fwdprobs[('stop',len(sentence)+1)] = logExpSumTrick(lst)
	return fwdprobs

#Computing backward probabilities
def calbackprobs(A,B,sentence,taglist):

	backprobs = {}
	
	for tags in taglist:
		for i in range(len(sentence) + 2):
			backprobs[(tags,i)] = 0
			
	#First Step (from stop to other tags)
	for tag in taglist:
		try:
			trans_prob = A['stop'][tag]
		except KeyError:
			trans_prob = math.log(0.01)
		backprobs[(tag,len(sentence))] = trans_prob
			
	
	#Next steps
	lst = []	
	for time in reversed(range(1,len(sentence))):
		for tag in taglist:
			lst = []
			for tagip1 in taglist:
				try:
					em_prob = B[tag][sentence[time][0]]
				except KeyError:
					em_prob = math.log(0.01)
				try:
					trans_prob = A[tagip1][tag]
				except KeyError:
					trans_prob = math.log(0.01)
				lst.append(backprobs[(tagip1,time+1)] + em_prob + trans_prob)
			backprobs[(tag,time)] = logExpSumTrick(lst)
			
			
	#Last step ( when you are at start )		
	lst = []
	for tag in taglist:
		try:
			em_prob = B[tag][sentence[1][0]]
		except KeyError:
			em_prob = math.log(0.01)
		try:
			trans_prob = A[tag]['start']
		except KeyError:
			trans_prob = math.log(0.01)
		lst.append(backprobs[(tag,1)] + trans_prob + em_prob)		
	backprobs[('start',0)] = logExpSumTrick(lst)
	return backprobs
	
	


	
def listofpostags(training):
	tagcount = {}	
	
	tagcount['start'] = 0
	tagcount['stop'] = 0
	
	for element in training:
		for pairs in element:
			tagcount[pairs[1]] = 0

	for element in training:
		tagcount['start'] = tagcount['start'] + 1
		tagcount['stop'] = tagcount['stop'] + 1
		for pairs in element:
			tagcount[pairs[1]] = tagcount[pairs[1]] + 1
	
	return tagcount
			
def logExpSumTrick(lst):
	largest = max(lst)
	sum = 0
	for prob in lst:
		sum += math.exp(prob-largest)
    	sum = math.log(sum)
    	sum = largest + sum
	
	return sum		
	
full_training=nltk.corpus.treebank.tagged_sents()[0:300]
#training_set1=full_training[0:1750]
#training_set2=full_training[1750:]
#test_set=nltk.corpus.treebank.tagged_sents()[3500:]



print("counting...")
(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(full_training)
#(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(training_set1)


#print sentence
taglist = listofpostags(full_training).keys()

#print taglist

print("Forward-Backward");

for sentence in full_training:
	Fprobs = calfwdprobs(tagtagcount_table,wrdtagcount_table,sentence,taglist)
	#print Fprobs
	Bprobs = calbackprobs(tagtagcount_table,wrdtagcount_table,sentence,taglist)
	#print Bprobs