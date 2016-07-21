import TitanicNew
import unittest

# Set up test class to test my model
class TestSVCModel(unittest.TestCase):
    # take the X_train and Y_train as created in TitanicNew
    def setUp(self):
        self.X_train = TitanicNew.X_train
        self.Y_train = TitanicNew.Y_train
    
    # Test 20 iterations that the prediction accuracy is always over 85%
    def test_prediction_percentage(self):
        self.assertTrue(all(0.85 < TitanicNew.get_prediction_accuracy(self.X_train, self.Y_train) for i in range(1, 20)))
        
if __name__ == '__main__':
    unittest.main()  