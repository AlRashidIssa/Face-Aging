import tensorflow as tf
import numpy as np
from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress the specific warning
import warnings
warnings.simplefilter('ignore')

class ImageProcessor:
    """
    Class for processing and augmenting images using TensorFlow.

    Attributes:
        augmentation_params (dict): Dictionary of augmentation parameters for ImageDataGenerator.
        images (np.ndarray): Array of images to preprocess.
        batch_size (int): Size of batches.
    Methods:
        __init__(self, augmentation_params: dict, images: np.ndarray, batch_size: int):
            Initialize the ImageProcessor with augmentation parameters, images, and batch size.
        
        preprocess_images(self) -> np.ndarray:
            Apply scaling to images to make the range between (0, 1).
        
        augment_images(self) -> tf.data.Dataset:
            Augment images using ImageDataGenerator and return a TensorFlow dataset.
    """

    def __init__(self, augmentation_params: dict, images: np.ndarray, batch_size: int) -> None:
        """
        Initialize the image processor with augmentation parameters, images, and batch size.

        Args:
            augmentation_params (dict): Dictionary of augmentation parameters for ImageDataGenerator.
            images (np.ndarray): Array of images to preprocess and augment.
            batch_size (int): Size of batches for augmentation.
        """
        self.datagen = ImageDataGenerator(**augmentation_params)
        self.images = images
        self.batch_size = batch_size

    def preprocess_images(self) -> np.ndarray:
        """
        Preprocess the images by scaling them between 0 and 1.

        Returns:
            np.ndarray: Preprocessed images.
        """
        return self.images / 255.0
    
    def augment_images(self) -> tf.data.Dataset:
        """
        Augment images using ImageDataGenerator.

        Returns:
            tf.data.Dataset: Dataset of augmented images.
        """
        # Create a generator that yields individual images
        def data_generator():
            for batch in self.datagen.flow(self.images, batch_size=self.batch_size):
                for img in batch:
                    yield img

        return tf.data.Dataset.from_generator(
            data_generator,
            output_signature=tf.TensorSpec(shape=self.images.shape[1:], dtype=tf.float32)
        )