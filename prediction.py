from numpy import ndarray
import tensorflow as tf
import numpy as np

class Prediction:
    """
    Class for loading a trained model and making predictions.

    Attributes:
        image (np.ndarray): The input image for prediction.
        train_model (str): Path to the trained model.

    Methods:
        __init__(image: np.ndarray, train_model: str): Initialize the Prediction class.
        load_model() -> tf.keras.Model: Load the trained model.
        preprocess() -> np.ndarray: Preprocess the image by scaling it between 0 and 1.
        predict() -> np.ndarray: Make predictions using the loaded model.
    """

    def __init__(self, image: np.ndarray, train_model: str) -> None:
        """
        Initialize the Prediction class with the given image and trained model path.

        Args:
            image (np.ndarray): The input image for prediction.
            train_model (str): Path to the trained model.
        """
        self.image = image
        self.train_model = train_model
        self.model = self.load_model()
        self.pr_image = self.preprocess()

    def load_model(self) -> tf.keras.Model:
        """
        Load the trained model from the specified path.

        Returns:
            tf.keras.Model: The loaded model.
        """
        model = tf.keras.models.load_model(self.train_model)
        return model
    
    def preprocess(self) -> np.ndarray:
        """
        Preprocess the image by scaling it between 0 and 1.

        Returns:
            np.ndarray: The preprocessed image.
        """
        return self.image / 255.0
    
    def predict(self) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Returns:
            np.ndarray: The generated images.
        """
        preprocessed_image = self.pr_image
        new_image = self.model.predict(np.expand_dims(preprocessed_image, axis=0))
        return new_image


import matplotlib.pylab as plt
# Example Usage
if __name__ == "__main__":
    # Load an example image
    (X_train, _), (_, _) = 0 , 0, 0, 0
    example_image = np.expand_dims(X_train[0], axis=-1)

    # Path to the trained model
    trained_model_path = 'path/to/your/trained/model.h5'

    # Initialize and use the Prediction class
    predictor = Prediction(image=example_image, train_model=trained_model_path)
    generated_image = predictor.predict()

    # Display the original and generated images
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(example_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Generated Image')
    plt.imshow(generated_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()
