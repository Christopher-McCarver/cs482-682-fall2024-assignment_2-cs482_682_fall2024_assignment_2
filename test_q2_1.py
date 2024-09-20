#!/usr/bin/env python3
from kmeans_clustering import MykmeansClustering
import unittest
import numpy as np

class TestMyLogisticRegression(unittest.TestCase):

	def test_basic_play_game_1(self):
		classifier = MykmeansClustering('dataset_q2.mat')
		clusters = classifier.model_fit()

		if clusters.shape[0] < 3:
			print("For this dataset a minimum of 3 clusters is expected")
		self.assertTrue(clusters.shape[0] >= 3)

	def test_basic_play_game_2(self):
		classifier = MykmeansClustering('dataset_q2.mat')
		clusters = classifier.model_fit()
		
		res = np.array([[1.95399466, 5.02557006],
						[3.04367119, 1.01541041],
						[6.03366736, 3.00052511]])

		err_large = False
		min_errs = []
		err_thresh = 0.9
		for i in range(res.shape[0]):
			dist = []
			for j in range(clusters.shape[0]):
				norm = np.linalg.norm(res[i,:]-clusters[j,:])
				dist.append(norm)
			min_ = min(dist)
			min_errs.append(min_)
			if min_>err_thresh:
				# print("large error at i={}, j={}, err = {}".format(i,j,min_))
				err_large = True
		# print("err large is {},  max min err = {}".format(err_large,max(min_errs)))
		self.assertTrue(not err_large)
		print('Expected result:\nclusters = {}'.format(res))
		print('Your result:\nclusters = {}'.format(clusters))

if __name__ == '__main__':
    unittest.main()