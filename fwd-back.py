import math
import prob1

#Computing forward probabilities
def calfwdprobs(a,b,utterance,taglist):

	fwd_probs = {}
	for tags in taglist:
		for i in range(10000):
			fwdprobs[tags][i] = 0

	for tags in taglist:
		fwdprobs['start'][0] = a['start'][tags] * b[tags][utterance[0][0]]
	for t in range(1,len(utterance)-1):
		for tags in taglist:
			sum = 0
			for s1 in taglist:
				sum  = sum + fwdprobs[s1][t-1] * a[s1][tags] * b[tags][utterance[t][0]]
			fwdprobs[tags][words] = sum
	sum = 0
	for tags in tagslist:
		sum = sum + fwdprobs[tags][len(utterance)]*a[tags][]		
	fwdprobs[][] = sum
	return fwdprobs


	
full_training=nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1=full_training[0:1750]
training_set2=full_training[1750:]
test_set=nltk.corpus.treebank.tagged_sents()[3500:]

print("counting...")
(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(full_training)
(wrdtagcount_table,tagtagcount_table) = prob1.calculateprobtables(training_set1)
