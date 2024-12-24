import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
import time
from load_data_cifer import load_cifar10_data  

data_dir = 'cifar-10-batches-py'  
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

pca = PCA(0.90)
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

class RBFNetwork:
    def __init__(self, num_centers, output_dim):
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.centers = None
        self.weights = None
        self.encoder = OneHotEncoder()

    def _rbf(self, x, center, sigma):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = self._rbf(x, c, self.sigma)
        return G

    def fit(self, X, y):
        y_one_hot = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()

        kmeans = KMeans(n_clusters=self.num_centers, random_state=0)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        self.sigma = np.mean(euclidean_distances(self.centers))

        G = self._calculate_interpolation_matrix(X)

        self.weights = np.dot(np.linalg.pinv(G), y_one_hot)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return np.argmax(predictions, axis=1)

num_centers = 100  

rbf_net = RBFNetwork(num_centers=num_centers, output_dim=10)
start_time = time.time()
rbf_net.fit(x_train_pca, y_train)
training_time = time.time() - start_time

start_time = time.time()
y_pred = rbf_net.predict(x_test_pca)
testing_time = time.time() - start_time

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("RBF Neural Network Results")
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

with open("results_report.txt", "w") as f:
    f.write("RBF Neural Network Results\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Testing time: {testing_time:.2f} seconds\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
