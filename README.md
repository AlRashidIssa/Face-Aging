Sure, here's a markdown file for GitHub:

```markdown
# Project Title

## Overview

This project consists of several Python scripts and classes for training a Generative Adversarial Network (GAN) model on a dataset of facial images to generate synthetic facial images.

The project includes the following components:

- **Ingest Data**: A class for loading images from a storage directory, including all subdirectories.
- **Image Processor**: A class for processing and augmenting images using TensorFlow.
- **Train Model**: A class for training the GAN model.
- **Prediction**: A class for loading a trained model and making predictions.
- **Evaluation Metrics**: A class for computing evaluation metrics for model predictions.

## Usage

### 1. Ingest Data

This component loads images from a directory and its subdirectories.

```python
from ingest_data import IngestData

# Initialize the IngestData class
data_ingestor = IngestData(path=main_directory, max_images=20000, target_size=input_shape[:2])

# Load images
images = data_ingestor.get_images()
```

### 2. Image Processor

This component preprocesses and augments images using TensorFlow.

```python
from image_processor import ImageProcessor

# Initialize the ImageProcessor class
data_augmentor = ImageProcessor(augmentation_params=augmentation_params, images=images, batch_size=batch_size)

# Preprocess images
x_train = data_augmentor.preprocess_images()

# Augment images
augmented_dataset = data_augmentor.augment_images()
```

### 3. Train Model

This component trains the GAN model.

```python
from model_utils import TrainModel

# Initialize the TrainModel class
trainer = TrainModel(input_shape=input_shape, epochs=epochs, batch_size=batch_size, x_train=x_train)

# Train the model
trainer.train(sample_interval=200)

# Save the trained model
trainer.save("version.h5")
```

### 4. Prediction

This component makes predictions using the trained model.

```python
from prediction import Prediction

# Initialize the Prediction class
predictor = Prediction(image=example_image, train_model=trained_model_path)

# Generate images
generated_image = predictor.predict()
```

## Contributors

- [Contributor 1](link-to-contributor1)
- [Contributor 2](link-to-contributor2)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Replace the placeholders such as `Project Title`, `Overview`, `Usage`, `Contributors`, `License`, etc., with appropriate content specific to your project. Additionally, provide links to the contributors' profiles and update the license information accordingly.