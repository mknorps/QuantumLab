import unittest
import numpy as np

from src import cost 
 



class CostTests(unittest.TestCase):

    '''
    def test_init(self):
        a = nn.SimpleNN(2,2)
        self.assertEqual(a.n,2)
        self.assertEqual(a.nn,2)
        self.assertEqual(a.A.shape,(2,))
        self.assertEqual(a.W1.shape,(2,3))
        self.assertEqual(a.W2.shape,(3,))

    def test_forward_propagate(self):
        sample = np.array([1,1,1]) # for 2nd order polynomial
        a = nn.SimpleNN(2,2)
        a.W1 = np.ones([2,3])
        a.W2 = np.ones(3)
        a.forward_propagate(sample, lambda x:x)
        try:
            np.testing.assert_array_equal(a.A, [3,3])
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a.A, [3,3]))
            res=False
        self.assertTrue(res)

    def test_forward_propagate_default(self):
        sample = np.array([1,1,1]) # for 2nd order polynomial
        a = nn.SimpleNN(2,2)
        a.W1 = np.zeros([2,3])
        a.forward_propagate(sample)
        try:
            np.testing.assert_array_equal(a.A, [0.5,0.5])
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a.A, [0.5,0.5]))
            res=False
        self.assertTrue(res)
    '''

if __name__=='__main__':
    unittest.main()
