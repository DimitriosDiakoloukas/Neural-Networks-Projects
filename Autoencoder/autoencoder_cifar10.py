import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from load_data_cifer import load_cifar10_data

class AutoencoderCifar(Model):
    def __init__(self):
        super(AutoencoderCifar, self).__init__()
        
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'),
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, (3, 3), strides=2, padding='same', activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

data_dir = 'cifar-10-batches-py'
(X_train, _), (_, _) = load_cifar10_data(data_dir)

X_train = X_train.astype("float32") / 255.0

# Prepare TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(64)

AutoencoderCifar = AutoencoderCifar()
AutoencoderCifar.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse')

num_epochs = 10
AutoencoderCifar.fit(train_dataset, epochs=num_epochs)

AutoencoderCifar.save_weights('AutoencoderCifar.h5')

test_dataset = train_dataset.take(1)  
test_inputs = next(iter(test_dataset))

reconstructed_outputs = AutoencoderCifar(test_inputs)

def imshow_and_save(img_batch, title, filename):
    img_batch = tf.clip_by_value(img_batch, 0.0, 1.0)
    grid = np.concatenate([np.concatenate(img_batch.numpy(), axis=1)], axis=0)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

imshow_and_save(test_inputs, "Original Images", "original_images.png")
imshow_and_save(reconstructed_outputs, "Reconstructed Images", "reconstructed_images.png")

print("Saved original and reconstructed images as 'original_images.png' and 'reconstructed_images.png'.")
