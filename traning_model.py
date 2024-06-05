import numpy as np
from train_model import TrainModel
from pr_process_data import ImageProcessor
from ingest_data import IngestData

import os
import logging
import tensorflow as tf
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set logging level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure absl logging to suppress INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

if __name__ == "__main__":
    # Parameters
    input_shape = (256, 256, 1)
    epochs = 1000
    batch_size = 1

    # Define augmentation parameters
    augmentation_params = {
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
        "zoom_range": 0.1,
        "horizontal_flip": True,
        "fill_mode": "nearest"
    }

    # Path to the main directory containing subdirectories with images
    main_directory = '/kaggle/input/face-againg-0/imdb'

    # Initialize and use the IngestData class with a limit of 20000 images and resize to 256x256
    data_ingestor = IngestData(path=main_directory, max_images=20000, target_size=input_shape[:2])
    images = data_ingestor.get_images()

    # Initialize ImageProcessor with augmentation parameters
    data_augmentor = ImageProcessor(augmentation_params=augmentation_params, images=images, batch_size=batch_size)
    x_train = data_augmentor.preprocess_images()
    augmented_dataset = data_augmentor.augment_images()

    # Detect and initialize TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    # Train the model within the strategy scope
    with strategy.scope():
        trainer = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=x_train)
        trainer.train(sample_interval=200)
        trainer.save("version.h5")
