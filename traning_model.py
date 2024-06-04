import numpy as np
from train_model import TrainModel
from pr_process_data import ImageProcessor
from ingest_data import IngestData

# Parameters
input_shape = (100,)
epochs = 10000
batch_size = 32




# Example Usage
if __name__ == "__main__":
    # Path to the directory containing images
    image_directory = 'path_to_images_directory'  # Update this with the actual path

    # Initialize and use the IngestData class
    data_ingestor = IngestData(path=image_directory)
    images = data_ingestor.get_images()

    # Define augmentation parameters
    augmentation_params = {
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "zoom_range": 0.1
    }

    # Initialize image processor
    image_processor = ImageProcessor(augmentation_params=augmentation_params)

    # Preprocess images
    preprocessed_images = image_processor.preprocess_images(images)

    # Initialize and train the GAN
    train_model = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=preprocessed_images)
    train_model.train(sample_interval=200)
