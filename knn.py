import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from load_data_cifer import load_cifar10_data

data_dir = 'cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

x_train_1D = x_train.reshape(x_train.shape[0], -1)
print(x_train_1D)
x_test_1D = x_test.reshape(x_test.shape[0], -1)
print(x_test_1D)

pca = PCA(n_components=100)
X_train_1D = pca.fit_transform(x_train_1D)
x_test_1D = pca.transform(x_test_1D)

knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train_1D, y_train)
y_pred_knn_1 = knn_1.predict(x_test_1D)
accuracy_knn_1 = accuracy_score(y_test, y_pred_knn_1)
print(f"KNN k=1 accuracy: {accuracy_knn_1}")

knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train_1D, y_train)
y_pred_knn_3 = knn_3.predict(x_test_1D)
accuracy_knn_3 = accuracy_score(y_test, y_pred_knn_3)
print(f"KNN k=3 accuracy: {accuracy_knn_3}")

centroid_clf = NearestCentroid()
centroid_clf.fit(X_train_1D, y_train)
y_pred_centroid = centroid_clf.predict(x_test_1D)
accuracy_centroid = accuracy_score(y_test, y_pred_centroid)
print(f"Nearest centroid accuracy: {accuracy_centroid}")