import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_recall_curve
from prediction import Prediction

class EvaluationMetrics(Prediction):
    """
    Class to compute evaluation metrics for model predictions.

    Attributes:
        image (np.ndarray): Input image for prediction.
        train_model (str): Path to the trained model.

    Methods:
        __init__: Initializes the EvaluationMetrics class.
        accuracy: Computes the accuracy of the model predictions.
        f1_score: Computes the F1 score of the model predictions.
        recall_score: Computes the recall score of the model predictions.
        precision_recall_curve: Computes the precision-recall curve of the model predictions.
    """

    def __init__(self, image: np.ndarray, train_model: str) -> None:
        """
        Initialize the EvaluationMetrics class.

        Args:
            image (np.ndarray): Input image for prediction.
            train_model (str): Path to the trained model.
        """
        super().__init__(image, train_model)

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy of the model predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the F1 score of the model predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: F1 score.
        """
        return f1_score(y_true, y_pred)

    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the recall score of the model predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Recall score.
        """
        return recall_score(y_true, y_pred)

    def precision_recall_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the precision-recall curve of the model predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Precision, recall, and threshold arrays.
        """
        return precision_recall_curve(y_true, y_pred)
