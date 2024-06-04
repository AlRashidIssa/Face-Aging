import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from typing import List

class IngestData:
    """
    Class for loading images from a storage directory.

    Attributes:
        path (str): The path to the directory containing images.

    Methods:
        __init__(self, path: str) -> None: Initialize the IngestData class with the given path.
        get_images(self) -> np.ndarray: Load and return images from the directory as a NumPy array.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the IngestData class with the given path.

        Args:
            path (str): The path to the directory containing images.
        """
        self.path = path

    def get_images(self) -> np.ndarray:
        """
        Load and return images from the directory as a NumPy array.

        Returns:
            np.ndarray: Array of images loaded from the directory.
        """
        image_list = []
        for filename in os.listdir(self.path):
            img_path = os.path.join(self.path, filename)
            if os.path.isfile(img_path):
                img = load_img(img_path, color_mode='grayscale')
                img_array = img_to_array(img)
                image_list.append(img_array)
        return np.array(image_list)

# Example Usage
if __name__ == "__main__":
    # Path to the directory containing images
    image_directory = 'path/to/your/image/directory'

    # Initialize and use the IngestData class
    data_ingestor = IngestData(path=image_directory)
    images = data_ingestor.get_images()

    # Display the first image
    if images.size > 0:
        import matplotlib.pyplot as plt
        plt.imshow(images[0].squeeze(), cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        print("No images found in the specified directory.")
