import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import time
from load_data_cifer import load_cifar10_data

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

data_dir = 'cifar-10-batches-py'
(x_train, _), (_, _) = load_cifar10_data(data_dir)

autoencoder = Autoencoder()

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError()
)

batch_size = 64
num_epochs = 10

start_time = time.time()

history = autoencoder.fit(
    x_train, x_train,
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle=True,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

def save_images(original, reconstructed, filename):
    n = 5
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.axis('off')
    for i in range(n):
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed[i])
        plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

test_batch = x_train[:5]
reconstructed_images = autoencoder.predict(test_batch)

save_images(test_batch, reconstructed_images, 'autoencoder_comparison_tf_cifar.png')

def calculate_accuracy(original, reconstructed):
    mse = tf.reduce_mean(tf.square(original - reconstructed)).numpy()
    accuracy = 1 - mse
    return accuracy

accuracy = calculate_accuracy(test_batch, reconstructed_images)

print(f"Training Time: {training_time:.2f} seconds")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Reconstruction Accuracy (1 - MSE): {accuracy:.4f}")

with open("autoencoder_results_tf_cifar.txt", "w") as f:
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Reconstruction Accuracy (1 - MSE): {accuracy:.4f}\n")
    f.write(f"Comparison Image Saved as: 'comparison_tf_fit.png'\n")
