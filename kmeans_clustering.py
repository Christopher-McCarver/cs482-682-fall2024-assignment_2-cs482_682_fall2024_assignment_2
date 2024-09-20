import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.data = None
        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.data = mat['X']
        
    def model_fit(self, n_clusters=3, max_iter=300):
        '''
        Initialize self.model here and execute kmeans clustering on the dataset.
        '''
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
        self.model.fit(self.data)
        cluster_centers = self.model.cluster_centers_
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.model.labels_, s=50, cmap='viridis')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
        plt.title(f'KMeans Clustering with {n_clusters} Clusters')
        plt.show()
        return cluster_centers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit()
    print(clusters_centers)
    