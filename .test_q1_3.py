#!/usr/bin/env python3
from logistic_regression import MyLogisticRegression
import unittest
import numpy as np

class TestMyLogisticRegression(unittest.TestCase):

	def test_basic_play_game_1(self):
		classifier = MyLogisticRegression('1')
		[accuracy, precision, recall, f1, support] = classifier.model_predict_logistic()

		ans = accuracy >=0.9 and precision[0] >= 0.90 and recall[0] >= 0.81 \
            	and f1[0] >= 0.85 and support[0] >=11 \
                and precision[1] >= 0.90 and recall[1] >= 0.94 \
            	and f1[1] >= 0.92 and support[1] >=19
		print('\nexpected results:')
		print('Accuracy: 0.90')
		print('class 0 | p r f1 sup = 0.90, 0.82, 0.86, 11.00')
		print('class 1 | p r f1 sup = 0.90, 0.95, 0.92, 19.00')
		print('Your results:')
		print('Accuracy: {}'.format(accuracy))
		print('class 0 | p r f1 sup = {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(precision[0], recall[0], f1[0], support[0]))
		print('class 1 | p r f1 sup = {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(precision[1], recall[1], f1[1], support[1]))
		self.assertTrue(ans)

if __name__ == '__main__':
    unittest.main()