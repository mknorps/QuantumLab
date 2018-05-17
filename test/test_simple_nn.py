import unittest
import numpy as np

from src import simple_nn as nn
 



class SimpleNNTests(unittest.TestCase):

    def test_init(self):
        a = nn.SimpleNN(2,2)
        self.assertEqual(a.n,2)
        self.assertEqual(a.nn,2)
        self.assertEqual(a.g(1),1)
        self.assertEqual(a.W1.shape,(2,3))
        self.assertEqual(a.W2.shape,(3,))

    def test_forward_propagate_default(self):
        sample = np.array([1,1,1]) # for 2nd order polynomial
        a = nn.SimpleNN(2,2)
        a.W1 = np.ones([2,3])
        a.W2 = np.ones(3)
        result, A1 = a.forward_propagate(sample)
        try:
            np.testing.assert_array_equal(A1, [3,3])
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(A1, [3,3]))
            res=False
        self.assertTrue(res)

    def test_forward_propagate_sigmoid(self):
        sample = np.array([1,1,1]) # for 2nd order polynomial
        a = nn.SimpleNN(2,2, act_function=lambda x: 1/(1+np.exp(-x)))
        a.W1 = np.zeros([2,3])
        a.W2 = np.zeros(3)
        result, A1 = a.forward_propagate(sample)
        try:
            np.testing.assert_array_equal(A1, [0.5,0.5])
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(A1, [0.5,0.5]))
            res=False
        self.assertTrue(res)
        self.assertEqual(result,0.5)


if __name__=='__main__':
    unittest.main()
