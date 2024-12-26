import numpy as np
from keras import layers, models
from keras.datasets import mnist
import matplotlib.pyplot as plt

def preprocess_data():
    (train_images, _), (_, _) = mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    train_images = np.reshape(train_images, (len(train_images), 28, 28, 1))
    return train_images

def build_encoder(latent_dim):
    encoder = models.Sequential()
    encoder.add(layers.Input(shape=(28, 28, 1)))
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(latent_dim, activation='relu'))
    return encoder

def build_decoder(latent_dim):
    decoder = models.Sequential()
    decoder.add(layers.Input(shape=(latent_dim,)))
    decoder.add(layers.Dense(7 * 7 * 16, activation='relu'))
    decoder.add(layers.Reshape((7, 7, 16)))
    decoder.add(layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'))
    return decoder

def plot_results(original_images, reconstructed_images, filename, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_loss(history, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_accuracy(original_images, reconstructed_images):
    mse = np.mean(np.square(original_images - reconstructed_images))
    return 1 - mse

latent_dim = 32
train_images = preprocess_data()
encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
autoencoder = models.Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(
    train_images, train_images,
    epochs=15,
    batch_size=128,
    shuffle=True,
    validation_split=0.2
)

encoded_images = encoder.predict(train_images)
decoded_images = decoder.predict(encoded_images)

plot_results(train_images, decoded_images, 'autoencoder_results.png')
plot_training_loss(history, 'training_loss.png')

accuracy = calculate_accuracy(train_images, decoded_images)

with open("autoencoder_results.txt", "w") as f:
    f.write(f"Reconstruction Accuracy (1 - MSE): {accuracy:.4f}\n")
    f.write("Training Loss and Validation Loss Plots saved as 'training_loss.png'.\n")
    f.write("Reconstructed and Original Images comparison saved as 'autoencoder_results.png'.\n")
