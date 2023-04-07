import unittest
from data_manipulation import load, save
import numpy as np
import pandas as pd
import os

class TestDataManipulation(unittest.TestCase):

    def test_load(self):
        # Test loading a CSV file
        file_path = 'data.csv'
        data = load(file_path)
        self.assertEqual(len(data), 10)
        
        # Test loading a pickle file
        file_path = 'data.pkl'
        data = load(file_path)
        self.assertEqual(len(data), 10)
        
        # Test loading a TXT file
        file_path = 'data.txt'
        data = load(file_path)
        self.assertEqual(len(data), 10)
        
    def test_save(self):
        # Test saving a CSV file
        file_path = 'data.csv'
        data = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
        save(data, file_path, 'csv')
        self.assertTrue(os.path.exists(file_path))
        
        # Test saving a pickle file
        file_path = 'data.pkl'
        data = np.random.randn(10, 4)
        save(data, file_path, 'pkl')
        self.assertTrue(os.path.exists(file_path))
        
        # Test saving a TXT file
        file_path = 'data.txt'
        data = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
        save(data, file_path, 'txt')
        self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main()
