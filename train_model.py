from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from model import GANModel

class TrainModel(GANModel):
    """
    Class to train the GAN model.
    """

    def __init__(self, input_shape: Tuple[int, ...], epochs: int, batch_size: int, x_train: np.ndarray) -> None:
        """
        Initialize the training process with specified parameters.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input noise vector for the generator.
            epochs (int): Number of epochs to train.
            batch_size (int): Size of each training batch.
            x_train (np.ndarray): Training data (images).
        """
        super().__init__(input_shape)
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_train = x_train

    def train(self, sample_interval: int = 200) -> None:
        """
        Train the GAN on the provided dataset.

        Args:
            sample_interval (int): Interval for saving generated image samples. Defaults to 200.
        """
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            # Train Discriminator
            idx = np.random.randint(0, self.x_train.shape[0], self.batch_size)
            real_imgs = self.x_train[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.input_shape[0]))
            gen_imgs = self.generator_.predict(noise)

            d_loss_real = self.discriminator_.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator_.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (self.batch_size, self.input_shape[0]))
            g_loss = self.gan_.train_on_batch(noise, valid)

            # Print progress
            if epoch % sample_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
                self.sample_images(epoch)

    def sample_images(self, epoch: int) -> None:
        """
        Generate and save images at specified epochs for visualization.

        Args:
            epoch (int): The current epoch number.
        """
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.input_shape[0]))
        gen_imgs = self.generator_.predict(noise)

        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()