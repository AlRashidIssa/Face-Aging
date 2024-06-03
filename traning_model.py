import numpy as np
from train_model import TrainModel
from pr_process import PrProcess 

# Example Usage
if __name__ == "__main__":
    # Load and preprocess dataset
    (X_train, _), (_, _) = PrProcess() # type: ignor
    X_train = (X_train - 127.5) / 127.5  # Normalize to [-1, 1]
    X_train = np.expand_dims(X_train, axis=-1)

    # Initialize and train GAN
    gan_trainer = TrainModel(input_shape=(100,), epochs=10000, batch_size=64, x_train=X_train)
    gan_trainer.train(sample_interval=200)