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
        b = np.array([np.ones(10), a[:,0],a[:,1]]).T
        try:
            np.testing.assert_array_equal(b,a_rescaled)
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(b, a_rescaled))
            res=False
        self.assertTrue(res)

    def test_0(self):
        a = np.array([[4,0],[0,2]])
        a_rescaled = pre.feature_scaling(a,0)
        b = np.array([[1,0],[1,2]])
        try:
            np.testing.assert_array_equal(a_rescaled, b)
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a_rescaled, b))
            res=False
        self.assertTrue(res)

    def test_1(self):
        a = np.array([[4,0],[0,2]])
        a_rescaled = pre.feature_scaling(a,1)
        b = np.array([[1,1,0],[1,-1,2]])
        try:
            np.testing.assert_array_equal(a_rescaled, b)
            res=True
        except AssertionError:
            print("{} in not equal to {}".format(a_rescaled, b))
            res=False
        self.assertTrue(res)
        

    def test_normalisation_0(self):
        a = 100*np.random.rand(100,2)
        a_rescaled = pre.feature_scaling(a,0)
        std_x1 = np.std(a_rescaled[:,0])
        self.assertAlmostEqual(std_x1,0)

    def test_normalisation_3(self):
        a = 10000* np.random.rand(100,2)
        a_rescaled = pre.feature_scaling(a,3)
        std_x1 = np.std(a_rescaled[:,1])
        std_x2 = np.std(a_rescaled[:,2])
        std_x3 = np.std(a_rescaled[:,3])
        self.assertAlmostEqual(std_x1,1)
        self.assertAlmostEqual(std_x2,1)
        self.assertAlmostEqual(std_x3,1)

class DataSplitTests(unittest.TestCase):

    def test_empty(self):
        a = pre.split_data([])
        self.assertDictEqual(a, {'test':[], 'train':[], 'valid':[]})

    def test_split(self):
        a = pre.split_data(np.random.rand(100,2))
        self.assertEqual(len(a['test']),20)
        self.assertEqual(len(a['train']),80)
        self.assertEqual(len(a['valid']),0)

    def test_reproducability(self):
        a = np.random.rand(100,2)
        b = pre.split_data(a)
        c = pre.split_data(a)
        for key in b.keys():
            try:
                np.testing.assert_array_equal(c[key], b[key])
                res=True
            except AssertionError:
                print("{} in not equal to {} for {}".format(c[key], b[key],key))
                res=False
            self.assertTrue(res)


if __name__=='__main__':
    unittest.main()
