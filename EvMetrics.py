import numpy as np
from evaluation_metrics import EvaluationMetrics

# Example usage
if __name__ == "__main__":
    # Example image and trained model path
    image = np.random.randn(100, 100)  # Example input image
    trained_model = "path/to/trained/model"

    # Initialize EvaluationMetrics object
    eval_metrics = EvaluationMetrics(image=image, train_model=trained_model)

    # Example true and predicted labels
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    # Compute and print accuracy
    accuracy = eval_metrics.accuracy(y_true, y_pred)
    print("Accuracy:", accuracy)

    # Compute and print F1 score
    f1 = eval_metrics.f1_score(y_true, y_pred)
    print("F1 Score:", f1)

    # Compute and print recall score
    recall = eval_metrics.recall_score(y_true, y_pred)
    print("Recall Score:", recall)

