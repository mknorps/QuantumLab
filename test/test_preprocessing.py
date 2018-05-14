import unittest
import numpy as np

from src import preprocessing as pre



class NormaliseArrayTests(unittest.TestCase):

    def test_constant(self):
        a = list(np.zeros(10))
        a_norm = list(pre._normalise_array(a))
        self.assertListEqual(a,a_norm)

    def test_ones(self):
        a = list(np.ones(10))
        a_norm = list(pre._normalise_array(a))
        b = list(np.zeros(10))
        self.assertListEqual(a_norm,b)

    def test_mean(self):
        a = [2,0]
        a_norm = list(pre._normalise_array(a))
        b = [1,-1]
        self.assertListEqual(a_norm,b)

    def test_std(self):
        a = [4,0]
        a_norm = list(pre._normalise_array(a))
        b = [1,-1]
        self.assertListEqual(a_norm,b)

class FeatureScalingTests(unittest.TestCase):

    def test_constant(self):
        a = np.zeros((10,2))
        a_rescaled = pre.feature_scaling(a,1)
        try:
            np.testing.assert_array_equal(a, a_rescaled)
            res=True
        except AssertionError:
            print("{} in not equal {}".format(a, a_rescaled))
            res=False
        self.assertTrue(res)

if __name__=='__main__':
    unittest.main()
