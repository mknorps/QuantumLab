import unittest
import numpy as np

from src import inout_nn as nn
 



class InOutNNTests(unittest.TestCase):

    def test_init(self):
        a = nn.InOutNN(2)
        self.assertEqual(a.n,2)
        self.assertEqual(a.W1.shape,(3,))

    def test_init(self):
        a = nn.InOutNN(2, init_random=True)
        self.assertEqual(a.n,2)
        self.assertEqual(a.W1.shape,(3,))

    def test_forward_propagate_zeros(self):
        sample = np.array([1,0,0]) # for 2nd order polynomial
        a = nn.InOutNN(2)
        a.W1 = np.array([0,0,0])
        A1 = a.forward_propagate(sample)
        self.assertEqual(A1,0 )

    def test_forward_propagate_ones(self):
        sample = np.array([1,1,1]) # for 2nd order polynomial
        a = nn.InOutNN(2)
        a.W1 = np.ones(3)
        A1 = a.forward_propagate(sample)
        self.assertEqual(A1,3 )

    def test_forward_propagate_linear(self):
        # y = 2x + 3
        sample = np.array([1,3]) # for 1st  order polynomial
        a = nn.InOutNN(1)
        a.W1 =np.array( [3,2])
        A1 = a.forward_propagate(sample)
        self.assertEqual(A1,9)

    def test_forward_propagate_3rdorder(self):
        # y = 2x**3+x-1
        sample = np.array([1,2,4,8]) # for 3rd order polynomial
        a = nn.InOutNN(1)
        a.W1 = np.array([-1,1,0,2])
        A1 = a.forward_propagate(sample)
        self.assertEqual(A1,17)

class InOutNNCostTests(unittest.TestCase):

    def test_cost_ones(self):
        sample = np.array([1,1]) 
        a = nn.InOutNN(1)
        a.W1 = np.ones(2)
        cost,_ = a.cost(np.array([sample]),np.array([0]))
        self.assertEqual(cost,2)

    def test_cost_linear(self):
        # out = 2x + 3
        # y = 2
        sample = np.array([1,3]) # for 1st  order polynomial
        a = nn.InOutNN(1)
        a.W1 = np.array([3,2])
        cost,dcost = a.cost(np.array([sample]),np.array([2]))
        try:
            np.testing.assert_array_equal(dcost, np.array([7.0,21.0]))
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(dcost, np.array([7.0,21.0])))
            res=False
        self.assertTrue(res)
        self.assertEqual(cost,24.5)

    def test_cost_more_samples(self):
        sample = np.array([[1,1],[1,2],[1,3]]) 
        a = nn.InOutNN(1)
        a.W1 = np.ones(2)
        cost,_ = a.cost(sample,np.array([0,0,0]).T)
        self.assertEqual(cost,29/6)


class InOutNNGradientDescentTests(unittest.TestCase):

    def test_gradient_descent_linear(self):
        # y = 2x + 3
        sample = np.array([np.ones(10000), np.random.rand(10000)]).T # for 1st  order polynomial
        y = np.array([2*x+3 for x in sample[:,1]])
        a = nn.InOutNN(1, init_random=True)
        a.gradient_descent(sample,y)
        try:
            np.testing.assert_array_equal(a.W1, [3,2])
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a.W1, [3,2]))
            res=False
        self.assertTrue(res)

if __name__=='__main__':
    unittest.main()
