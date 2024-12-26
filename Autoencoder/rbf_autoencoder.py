from keras import layers, models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from load_data_cifer import load_cifar10_data

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

def build_autoencoder(input_dim, latent_dim):
    autoencoder = models.Sequential()
    autoencoder.add(layers.Input(shape=(input_dim,)))
    autoencoder.add(layers.Dense(512, activation='relu'))
    autoencoder.add(layers.Dense(latent_dim, activation='relu'))
    autoencoder.add(layers.Dense(512, activation='relu'))
    autoencoder.add(layers.Dense(input_dim, activation='sigmoid'))
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

data_dir = 'cifar-10-batches-py'  
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

latent_dim = 128
autoencoder = build_autoencoder(x_train_flat.shape[1], latent_dim)
autoencoder.fit(x_train_flat, x_train_flat, epochs=10, batch_size=128, shuffle=True)

encoder = models.Sequential(autoencoder.layers[:2])  
x_train_latent = encoder.predict(x_train_flat)
x_test_latent = encoder.predict(x_test_flat)

num_centers = 100
rbf_net = RBFNetwork(num_centers=num_centers, output_dim=10)
rbf_net.fit(x_train_latent, y_train)
y_pred = rbf_net.predict(x_test_latent)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

with open("results_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix, separator=', '))
