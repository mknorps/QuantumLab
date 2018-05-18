import unittest
import numpy as np

from src import inout_nn as nn
 

class InOutNNMyPolyfitTests(unittest.TestCase):

    def test_polyfit(self):
        fit = nn.my_polyfit(np.array([[1,1],[0,0]]))
        try:
            np.testing.assert_array_equal(fit, np.array([0,1]))
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(fit, np.array([0,1])))
            res=False
        self.assertTrue(res)

    def test_polyfit_quadratic(self):
        fit = nn.my_polyfit(np.array([[1,1],[0,0],[-1,1]]))
        try:
            np.testing.assert_array_equal(fit, np.array([0,0,1]))
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(fit, np.array([0,0,1])))
            res=False
        self.assertTrue(res)

    def test_polyfit_3rd(self):
        fit = nn.my_polyfit(np.array([[1,2],[0,1],[-1,0],[2,9]]))
        try:
            np.testing.assert_array_equal(fit, np.array([1,0,0,1]))
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(fit, np.array([1,0,0,1])))
            res=False
        self.assertTrue(res)

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

    def test_dcost_constant(self):
        # y = 300
        sample = np.array([np.ones(10000)]).T # for 1st  order polynomial
        y = 300*np.ones(10000) 
        a = nn.InOutNN(0)
        a.W1=np.array([1000])
        cost,dcost = a.cost(sample,y)
        self.assertEqual(dcost, np.array([700]))
        self.assertEqual(cost, 245000)
'''
class InOutNNGradientDescentTests(unittest.TestCase):

    def test_gradient_descent_constant(self):
        # y = 300
        sample = np.array([np.ones(10000)]).T # for 0  order polynomial
        y = 300*np.ones(10000) 
        a = nn.InOutNN(0, init_random=True)
        a.W1=np.array([1000])
        a.gradient_descent(sample,y, alpha=0.7, tol=0.0000001 )
        self.assertAlmostEqual(a.W1[0], 300, places=3)

    def test_gradient_descent_linear(self):
        # y = 2000*x - 300
        sample = np.array([np.ones(10000), np.random.rand(10000)]).T # for 1st  order polynomial
        y = np.array([2000*x+300 for x in sample[:,1]])
        a = nn.InOutNN(1, init_random=True)
        a.gradient_descent(sample,y, alpha=1, itmax = 200, tol=0.000001, verbose=False)
        try:
            np.testing.assert_array_almost_equal(a.W1, [300,2000], decimal=2)
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a.W1, [300,2000]))
            res=False
        self.assertTrue(res)

    def test_gradient_descent_quadratic(self):
        # y = -2000*x**2 + 545*x + 300
        random = np.random.rand(10000)
        sample = np.array([np.ones(10000), random, random**2]).T # for 2nd  order polynomial
        y = np.array([-2000*x**2 + 545*x +300 for x in sample[:,1]])
        a = nn.InOutNN(2, init_random=True)
        a.gradient_descent(sample,y, alpha=1.3, itmax = 3000, tol=0.000001, verbose=False)
        try:
            np.testing.assert_array_almost_equal(a.W1, [300,545,-2000], decimal=1)
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a.W1, [300,545,-2000]))
            res=False
        self.assertTrue(res)

    def test_gradient_descent_quadratic2(self):
        # y = 2*x**2 -545*x + 300
        random = np.random.rand(10000)
        sample = np.array([np.ones(10000), random, random**2]).T # for 2nd  order polynomial
        y = np.array([2*x**2 -545*x +300 for x in sample[:,1]])
        a = nn.InOutNN(2, init_random=True)
        a.gradient_descent(sample,y, alpha=1.3, itmax = 2500, tol=0.000001, verbose=False)
        try:
            np.testing.assert_array_almost_equal(a.W1, [300,-545,2], decimal=1)
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a.W1, [300,-545,2]))
            res=False
        self.assertTrue(res)

'''        

if __name__=='__main__':
    unittest.main()
