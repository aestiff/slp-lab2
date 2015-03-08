'''
Created on Mar 7, 2015

@author: stiff
'''
import unittest
from ForwardBackward import *

class Test(unittest.TestCase):


    def setUp(self):
        self.probs = (np.asarray(range(9))+1)*0.1
        self.probs = np.reshape(self.probs, (3,3))
        self.logProbs = np.log(self.probs)
        return


    def tearDown(self):
        pass


    def testLogExpSum(self):
        assert np.sum(np.log(np.sum(self.probs,1,keepdims=True))- logExpSumTrick(self.logProbs, 1)) < 0.001, "Difference: %r" % np.max(np.log(np.sum(self.probs,1,keepdims=True))- logExpSumTrick(self.logProbs, 1))
        assert np.sum(np.log(np.sum(self.probs,0,keepdims=True))- logExpSumTrick(self.logProbs, 0)) < 0.001, "Difference: %r" % np.max(np.log(np.sum(self.probs,0,keepdims=True))- logExpSumTrick(self.logProbs, 0))
        assert np.sum(np.log(np.sum(self.probs,keepdims=True))- logExpSumTrick(self.logProbs)) < 0.001, "Difference: %r" % np.max(np.log(np.sum(self.probs,keepdims=True))- logExpSumTrick(self.logProbs))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()