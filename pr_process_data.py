import tensorflow as tf
import numpy as np

from typing import Tuple, List
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

class ImageProcessor:
    """
    Class for processing and augmenting images using TensorFlow.

    Atrebutes:
            augmentation_params (dict): Dictionary of augmentation parameters for ImageDataGenerator.

            images (np.ndarray): Array of images to preprocess.

            batch_size (int): Size of batches.
    Methods:
        __init__(self, augmentiation_params: dict):
            Initialize all atrebutes and methodes.
        
        preprocess_image(self, images: np.ndarray) -> np.ndarray:
            for appling scaling for image make the reange between (0,1)
        
        augment_images(self): -> tf.data.Dataset:
            Augment images using ImageDataGenerator.

    """

    def __init__(self, augmentation_params: dict, images: np.ndarray, batch_size: int) -> None:
        """
        Initialize the image procesor with augmentation parameters.
        """
        self.datagen = ImageDataGenerator(**augmentation_params)
        self.images = images
        self.batch_size = batch_size

    def preprocess_images(self) -> np.ndarray:
        """
        Preprocess the image by scaling them between 0 and 1.

        Args:
            self.images (np.ndarray)

        Return:
            np.ndarray(): Preprocessed images.
        """

        return self.images / 255.0
    
    def augment_images(self) -> tf.data.Dataset:
        """
        Augment images using ImageDataGenerator.

        Args:
            self.images (np.ndarray): Array of images to augment.
            self.batch_size (int): Size of batches.

        Returns:
            tf.data.Dataset: Dataset of augmented images.
        """
        return tf.data.Dataset.from_generator(
            lambda: self.datagen.flow(self.images, batch_size=self.batch_size),
            output_signature=tf.TensorSpec(shape=self.images.shape[1:], dtype=tf.float32)
        )
