import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple

# Suppress the specific warning
import warnings
warnings.simplefilter('ignore')

class GANModel:
    """
    A class representing a Generative Adversarial Network (GAN).

    This class includes a Generator and a Discriminator.

    Attributes:
        input_shape (Tuple[int, int, int]): Shape of the input images for the generator.
    """

    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        """
        Initializes the GAN model.

        Args:
            input_shape (Tuple[int, int, int]): Shape of the input images for the generator.
        """
        self.input_shape = input_shape

        # Initialize the models
        self.discriminator_ = self.build_discriminator()
        self.generator_ = self.build_generator()
        self.gan_ = self.build_gan()

        # Compile the models
        self.compile_models()

    def build_generator(self) -> Model:
            """
            Builds the Generator model.

            The Generator takes a noise vector and transforms it into an image.

            Returns:
            tf.keras.Model: The Generator model.
            """
            model = tf.keras.Sequential()
            model.add(layers.Input(shape=(100,)))
            model.add(layers.Dense(16*16*512, use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Reshape((16, 16, 512)))
            model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            
            model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
            model.add(layers.Reshape((256, 256, 1)))  # Resize to match discriminator input shape

            return model
    
    def build_discriminator(self) -> Model:
        """
        Builds the Discriminator model.

        The Discriminator classifies whether an image is real or fake.

        Returns:
            tf.keras.Model: The Discriminator model.
        """
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def build_gan(self) -> Model:
        """
        Builds the GAN model.

        The GAN takes noise as input, generates an image, and then classifies it using the Discriminator.

        Returns:
            tf.keras.Model: The combined GAN model.
        """
        model = tf.keras.Sequential()
        model.add(self.generator_)
        model.add(self.discriminator_)

        return model

    def compile_models(self) -> None:
        """
        Compiles the Discriminator and GAN models with appropriate loss functions and optimizers.
        """
        optimizer = Adam(0.0002, 0.5)

        # Compile Discriminator
        self.discriminator_.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # The Discriminator should not be trained when training the GAN model
        self.discriminator_.trainable = False

        # Compile GAN
        self.gan_.compile(loss='binary_crossentropy', optimizer=optimizer)