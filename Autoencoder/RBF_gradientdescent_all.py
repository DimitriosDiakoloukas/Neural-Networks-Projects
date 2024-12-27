import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
from load_data_cifer import load_cifar10_data


class RBFLayer(layers.Layer):
    def __init__(self, units, initial_centers=None, gamma=1.0, is_trainable=True, name='rbflayer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.gamma = gamma
        self.initial_centers = initial_centers
        self.is_trainable = is_trainable

    def build(self, input_shape):
        if self.initial_centers is not None:
            initializer = tf.constant_initializer(self.initial_centers)
        else:
            initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)

        self.centers = self.add_weight(
            shape=(self.units, input_shape[-1]),
            initializer=initializer,
            trainable=self.is_trainable,
            name='centers'
        )

        self.gamma = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.gamma),
            trainable=False,
            name='gamma'
        )

        super().build(input_shape)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        squared_distances = tf.reduce_sum(tf.square(diff), axis=-1)
        outputs = tf.exp(-self.gamma * squared_distances)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


def train_rbfnn_model(X_train, Y_train, X_test, Y_test, is_trainable, gamma, num_centers, num_classes, input_shape_, epochs, random_centers=False):
    if random_centers is False:
        kmeans = KMeans(n_clusters=num_centers, random_state=42)
        kmeans.fit(X_train)
        initial_centers = np.array(kmeans.cluster_centers_)
    else:
        initial_centers = None

    model = models.Sequential()
    model.add(layers.Dense(input_shape_, activation='relu', input_shape=(input_shape_,)))
    model.add(RBFLayer(units=num_centers, initial_centers=initial_centers, gamma=gamma, is_trainable=is_trainable))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), batch_size=64, verbose=1)
    end_time = time.time()
    training_time = end_time - start_time

    return model, history, training_time


data_dir = 'cifar-10-batches-py'
(X_train, y_train), (X_test, y_test) = load_cifar10_data(data_dir)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = encoder.transform(y_test.reshape(-1, 1))

num_centers_list = [100, 300, 500]
gamma = 1.0
num_classes = 10
input_dim = 300
epochs = 50
is_trainable = True

os.makedirs('results', exist_ok=True)

for num_centers in num_centers_list:
    for random_centers in [False, True]:
        centers_type = "Random" if random_centers else "KMeans"
        model, history, training_time = train_rbfnn_model(
            X_train_pca, y_train_oh, X_test_pca, y_test_oh,
            is_trainable=is_trainable, gamma=gamma,
            num_centers=num_centers, num_classes=num_classes,
            input_shape_=input_dim, epochs=epochs, random_centers=random_centers
        )

        y_pred = np.argmax(model.predict(X_test_pca), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        results_text = (
            f"Number of Centers: {num_centers}\n"
            f"Centers Initialization: {centers_type}\n"
            f"Training Time: {training_time:.2f} seconds\n"
            f"Accuracy: {accuracy * 100:.2f}%\n"
            f"Classification Report:\n{report}\n"
        )

        results_file = f'results/rbfnn_results_{centers_type}_{num_centers}.txt'
        with open(results_file, 'w') as file:
            file.write(results_text)

        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss ({centers_type}, Centers: {num_centers})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/training_validation_loss_{centers_type}_{num_centers}.png')

        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy ({centers_type}, Centers: {num_centers})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'results/training_validation_accuracy_{centers_type}_{num_centers}.png')

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
        plt.title(f'Confusion Matrix ({centers_type}, Centers: {num_centers})')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(f'results/confusion_matrix_{centers_type}_{num_centers}.png')
