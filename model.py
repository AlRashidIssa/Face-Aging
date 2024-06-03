import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from typing import Tuple

class GANModel:
    """
    Class to define and handle the components of a Generative Adversarial Network (GAN).
    This GAN consists of a Generator and a Discriminator.
    """

    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        """
        Initialize the GAN model with the specified input shape.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input noise vector for the generator.
        """
        self.input_shape = input_shape

        # Initialize the models
        self.discriminator_ = self.discriminator()
        self.generator_ = self.generator()
        self.gan_ = self.gan()

        # Compile the models
        self.compile_models()

    def generator(self) -> Model:
        """
        Build the Generator model.
        The Generator takes a noise vector and transforms it into an image.

        Returns:
            model (tf.keras.Model): The Generator model.
        """
        model = tf.keras.Sequential([
            layers.Dense(7*7*256, use_bias=False, input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Reshape((7, 7, 256)),

            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])

        return model
    
    def discriminator(self) -> Model:
        """
        Build the Discriminator model.
        The Discriminator classifies whether an image is real or fake.

        Returns:
            model (tf.keras.Model): The Discriminator model.
        """
        model = tf.keras.Sequential([
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def gan(self) -> Model:
        """
        Build the GAN model by combining the Generator and Discriminator.
        The GAN takes noise as input, generates an image, and then classifies it using the Discriminator.

        Returns:
            model (tf.keras.Model): The combined GAN model.
        """
        model = tf.keras.Sequential()
        model.add(self.generator_)
        model.add(self.discriminator_)

        return model

    def compile_models(self) -> None:
        """
        Compile the Discriminator and GAN models with appropriate loss functions and optimizers.
        """
        optimizer = Adam(0.0002, 0.5)

        # Compile Discriminator
        self.discriminator_.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # The Discriminator should not be trained when training the GAN model
        self.discriminator_.trainable = False

        # Compile GAN
        self.gan_.compile(loss='binary_crossentropy', optimizer=optimizer)
