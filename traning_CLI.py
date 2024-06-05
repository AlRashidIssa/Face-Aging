import os 
import logging
import argparse
import tensorflow as tf
import warnings

from ingest_data import IngestData
from image_processor import ImageProcessor
from model_utils import TrainModel

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set logging level to suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def train_model(input_shape, epochs, batch_size, main_directory, accelerator=None):
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

    # Initialize and use the IngestData class to load and preprocess images
    data_ingestor = IngestData(path=main_directory, max_images=20000, target_size=input_shape[:2])
    images = data_ingestor.get_images()

    # Initialize ImageProcessor with augmentation parameters and loaded images
    data_processor = ImageProcessor(augmentation_params=augmentation_params, images=images, batch_size=batch_size)
    x_train = data_processor.preprocess_images()
    augmented_dataset = data_processor.augment_images()

    if accelerator == 'TPU':
        # Initialize TPU and distribute training
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        # Train the model within the strategy scope
        with strategy.scope():
            trainer = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=x_train)
            trainer.train(sample_interval=200)
            trainer.save_model("version.h5")
    elif accelerator == 'GPU':
        # Check and utilize GPU for training
        if tf.config.experimental.list_physical_devices('GPU'):
            print("Using GPU for training.")
            trainer = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=x_train)
            trainer.train(sample_interval=200)
            trainer.save_model("version.h5")
        else:
            print("No GPU available, falling back to CPU training.")
            trainer = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=x_train)
            trainer.train(sample_interval=200)
            trainer.save_model("version_0.h5")
    else:
        # Train model on CPU by default
        trainer = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=x_train)
        trainer.train(sample_interval=200)
        trainer.save_model("version_0.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[256, 256, 1], help='Shape of the input images')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--main_directory', type=str, required=True, help='Main directory containing image data')
    parser.add_argument('--accelerator', type=str, choices=['TPU', 'GPU'], help='Accelerator for training (TPU or GPU)')

    args = parser.parse_args()

    input_shape = tuple(args.input_shape)
    epochs = args.epochs
    batch_size = args.batch_size
    main_directory = args.main_directory
    accelerator = args.accelerator

    train_model(input_shape, epochs, batch_size, main_directory, accelerator)
