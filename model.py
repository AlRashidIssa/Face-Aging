import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers # type: ignore
from ten

from typing import List

class GnneratorModel:
    """
    
    """

    def __init__(self, input_shape: List[int, int, int]) -> None:
        self.model = self.build_model()

    
    def build_model(self) -> tf.keras.Model: