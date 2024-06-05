import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
from typing import List
import warnings
from PIL import Image

# Suppress the specific warning
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class IngestData:
    """
    Class for loading images from a storage directory, including all subdirectories.

    Attributes:
        path (str): The path to the directory containing images.
        max_images (int): The maximum number of images to load.
        target_size (tuple): The target size to resize images to.

    Methods:
        __init__(self, path: str, max_images: int, target_size: tuple) -> None: Initialize the IngestData class with the given path, max_images, and target_size.
        get_images(self) -> np.ndarray: Load and return images from the directory and subdirectories as a NumPy array.
    """

    def __init__(self, path: str, max_images: int = None, target_size: tuple = (224, 224)) -> None:
        """
        Initialize the IngestData class with the given path, max_images, and target_size.

        Args:
            path (str): The path to the directory containing images.
            max_images (int, optional): The maximum number of images to load. If None, load all images.
            target_size (tuple, optional): The target size to resize images to. Defaults to (224, 224).
        """
        self.path = path
        self.max_images = max_images
        self.target_size = target_size

    def get_images(self) -> np.ndarray:
        """
        Load and return images from the directory and all subdirectories as a NumPy array.

        Returns:
            np.ndarray: Array of images loaded from the directory and subdirectories.
        """
        image_list = []
        all_files = []

        # Collect all image file paths
        for root, _, files in os.walk(self.path):
            for filename in files:
                img_path = os.path.join(root, filename)
                if os.path.isfile(img_path):
                    all_files.append(img_path)

        # Shuffle the file list and select a subset if max_images is specified
        if self.max_images:
            random.shuffle(all_files)
            all_files = all_files[:self.max_images]

        # Load the images
        for img_path in all_files:
            try:
                img = load_img(img_path, color_mode='grayscale', target_size=self.target_size)
                img_array = img_to_array(img)
                image_list.append(img_array)
            except IOError as e:
                print(f"Error loading image {img_path}: {e}")
            except Exception as e:
                print(f"Unhandled error occurred while loading image {img_path}: {e}")

        print("Succeeded Ingest Data")
        return np.array(image_list)