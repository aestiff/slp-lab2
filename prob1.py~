import math
import nltk
from nltk.corpus import treebank


def calculateprobtables(training):

	tagcount = {} 
	wrdtagcount_table = {}
	tagtagcount_table = {}

	tagcount['start'] = 0
	tagcount['stop'] = 0

	for element in training:
		for pairs in element:
			wrdtagcount_table[(pairs[0],pairs[1])] = 0
			wrdtagcount_table[('UNK',pairs[1])] = 0.001
			tagcount[pairs[1]] = 0


	for element in training:
		tagcount['start'] = tagcount['start'] + 1
		tagcount['stop'] = tagcount['stop'] + 1
		for pairs in element:
			wrdtagcount_table[(pairs[0],pairs[1])] = wrdtagcount_table[(pairs[0],pairs[1])] + 1
			tagcount[pairs[1]] = tagcount[pairs[1]] + 1
		


	for element in training:
		for index in range(len(element)):
			if(index==0):
				tagtagcount_table[(element[0][1],'start')] = 0
			elif(index == (len(element)-1)):
				tagtagcount_table[('stop',element[index][1])] = 0
			else:
				tagtagcount_table[(element[index][1],element[index-1][1])] = 0
	

	for element in training:
		for index in range(len(element)):
			if(index==0):
				tagtagcount_table[(element[0][1],'start')] = tagtagcount_table[(element[0][1],'start')] + 1
			elif(index == (len(element)-1)):
				tagtagcount_table[('stop',element[index][1])] = tagtagcount_table[('stop',element[index][1])] + 1
			else:
				tagtagcount_table[(element[index][1],element[index-1][1])] = tagtagcount_table[(element[index][1],element[index-1][1])] + 1





	#Calculate P(W_i | T_i) = C(W_i, T_i) / C(T_i)  
	for (words,tags) in wrdtagcount_table.keys():
		if(words == "UNK"):
			continue
		wrdtagcount_table[(words,tags)] = - math.log1p(wrdtagcount_table[(words,tags)]) + math.log1p(tagcount[tags])
		


		
	#Calculate P(T_i | T_i-1) = C(T_i, T_i-1) / C(T_i-1)
	for (tagi,tagim1) in tagtagcount_table.keys():
		tagtagcount_table[(tagi,tagim1)] = - math.log1p(tagtagcount_table[(tagi,tagim1)]) + math.log1p(tagcount[tagim1])

	return (wrdtagcount_table,tagtagcount_table)



full_training=nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1=full_training[0:1750]
training_set2=full_training[1750:]
test_set=nltk.corpus.treebank.tagged_sents()[3500:]


(wrdtagcount_table,tagtagcount_table) =	calculateprobtables(full_training)
(wrdtagcount_table,tagtagcount_table) =	calculateprobtables(training_set1)

				 	
