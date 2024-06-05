import matplotlib.pyplot as plt
import numpy as np
from evaluation_metrics import EvaluationMetrics

class Visualization(EvaluationMetrics):
    """
    Visualization class for plotting various aspects of GAN training and evaluation.
    
    Attributes:
        sample_interval (int): Interval for sampling and visualizing generated images.
        history (dict): Dictionary to keep track of generator and discriminator loss history.
        metrics (dict): Dictionary to keep track of evaluation metrics history.
    
    Methods:
        plot_generated_images(epoch, examples=10): Plots generated images at a given epoch.
        plot_loss(): Plots the loss of the generator and discriminator over epochs.
        plot_metrics(): Plots evaluation metrics over epochs.
        update_metrics(epoch, y_true, y_pred): Updates the evaluation metrics after each epoch.
        save_plots(file_path): Saves the generated plots to a specified file path.
    """
    
    def __init__(self, image: np.ndarray, train_model: str, sample_interval=100):
        """
        Initialize the Visualization class.

        Args:
            image (np.ndarray): Input image for prediction.
            train_model (str): Path to the trained model.
            sample_interval (int): Interval for sampling and visualizing generated images.
        """
        super().__init__(image, train_model)
        self.model = self.load_model()
        self.sample_interval = sample_interval
        self.history = {'generator_loss': [], 'discriminator_loss': []}
        self.metrics = {'accuracy': [], 'f1_score': [], 'recall': []}
    
    def plot_generated_images(self, epoch, examples=10):
        """
        Plots generated images at a given epoch.
        
        Args:
            epoch (int): The current epoch number.
            examples (int): Number of examples to generate and plot.
        """
        noise = np.random.normal(0, 1, (examples, self.model.input_shape[1]))
        generated_images = self.model.generator.predict(noise)
        
        plt.figure(figsize=(10, 10))
        for i in range(examples):
            plt.subplot(1, examples, i + 1)
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')
        
        plt.suptitle(f'Generated Images at Epoch {epoch}')
        plt.show()
    
    def plot_loss(self):
        """
        Plots the loss of the generator and discriminator over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['generator_loss'], label='Generator Loss')
        plt.plot(self.history['discriminator_loss'], label='Discriminator Loss')
        plt.title('GAN Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def plot_metrics(self):
        """
        Plots evaluation metrics over epochs.
        """
        plt.figure(figsize=(10, 5))
        for metric, values in self.metrics.items():
            plt.plot(values, label=metric)
        plt.title('Evaluation Metrics Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.show()
    
    def update_metrics(self, epoch, y_true, y_pred):
        """
        Updates the evaluation metrics after each epoch.
        
        Args:
            epoch (int): The current epoch number.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        self.metrics['accuracy'].append(self.accuracy(y_true, y_pred))
        self.metrics['f1_score'].append(self.f1_score(y_true, y_pred))
        self.metrics['recall'].append(self.recall_score(y_true, y_pred))
        print(f"Epoch {epoch} - Accuracy: {self.metrics['accuracy'][-1]}, F1 Score: {self.metrics['f1_score'][-1]}, Recall: {self.metrics['recall'][-1]}")

    def save_plots(self):
        """
        Save the plots to a specified file path.

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['generator_loss'], label='Generator Loss')
        ax1.plot(self.history['discriminator_loss'], label='Discriminator Loss')
        ax1.set_title('GAN Loss Over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot metrics
        for metric, values in self.metrics.items():
            ax2.plot(values, label=metric)
        ax2.set_title('Evaluation Metrics Over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric Value')
        ax2.legend()
        
        plt.savefig("PlotsEvaluation.png")
        plt.close()
        print(f"Plots saved to {"PlotsEvaluation.png"}")

