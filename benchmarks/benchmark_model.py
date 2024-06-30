"""
This module provides a BenchmarkModel class for benchmarking machine learning models.

The BenchmarkModel class supports Keras and TensorFlow Lite models, allowing for
custom preprocessing, inference, and evaluation with user-defined metrics.

Example usage:
    import numpy as np
    from benchmark_model import BenchmarkModel, ModelType

    # Prepare your data
    X = np.random.rand(1000, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 10, 1000)

    # Define a custom preprocessing function
    def preprocess(data):
        return data / 255.0

    # Define custom metrics
    def custom_metric(predictions, labels):
        return np.mean(np.argmax(predictions, axis=1) == labels)

    # Create a BenchmarkModel instance
    model = BenchmarkModel(
        name="MyModel",
        model_path="path/to/my_model.h5",
        model_type=ModelType.KERAS,
        dataset=(X, y),
        preprocess_func=preprocess,
        metrics={"custom_accuracy": custom_metric}
    )

    # Preprocess data
    preprocessed_data = model.preprocess_data(X)

    # Run inference
    predictions = model.run_inference(preprocessed_data)

    # Evaluate the model
    results = model.evaluate(predictions, y)
    print(f"Evaluation results: {results}")

    # Get model size
    size = model.get_size()
    print(f"Model size: {size:.2f} MB")
"""

from typing import Tuple, Callable, Dict, Optional
from enum import Enum
import os
import numpy as np
import tensorflow as tf

class ModelType(Enum):
    """Enumeration of supported model types."""
    KERAS = "keras"
    TFLITE = "tflite"

class BenchmarkModel:
    """
    A class for benchmarking machine learning models.

    This class supports Keras and TensorFlow Lite models, allowing for
    custom preprocessing, inference, and evaluation with user-defined metrics.
    """

    def __init__(self, name: str, model_path: str, model_type: ModelType, 
                 test_dataset: Tuple[np.ndarray, np.ndarray], 
                 preprocess_func: Callable[[np.ndarray], np.ndarray] = None,
                 metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None):
        """
        Initialize a BenchmarkModel instance.

        Args:
            name (str): A string identifier for the model.
            model_path (str): Path to the model file.
            model_type (ModelType): Type of the model (KERAS or TFLITE).
            dataset (Tuple[np.ndarray, np.ndarray]): A tuple containing input data (X) and labels (y).
            preprocess_func (Callable[[np.ndarray], np.ndarray], optional): A function to preprocess input data.
            metrics (Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional): A dictionary of metric functions for model evaluation.
        """
        self.name = name
        self.path = model_path
        self.type = model_type
        self.X_test, self.y_test = test_dataset
        self.preprocess_func = preprocess_func or (lambda x: x)
        self.metrics = metrics if metrics is not None else {"accuracy": self._default_accuracy}
        self._load_model()

    def _load_model(self):
        """
        Load the model based on its type.

        Raises:
            ValueError: If the model type is not supported.
        """
        if self.type == ModelType.KERAS:
            self.model = tf.keras.models.load_model(self.path)
        elif self.type == ModelType.TFLITE:
            self.model = tf.lite.Interpreter(model_path=self.path)
            self.model.allocate_tensors()
        else:
            raise ValueError(f"Unsupported model type: {self.type}")

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing function to the input data.

        Args:
            data (np.ndarray): Input data to preprocess.

        Returns:
            np.ndarray: Preprocessed data.
        """
        return self.preprocess_func(data)

    def run_inference(self, data: np.ndarray) -> np.ndarray:
        """
        Run inference on the input data using the loaded model.

        Args:
            data (np.ndarray): Input data for inference.

        Returns:
            np.ndarray: Model's predictions.
        """
        if self.type == ModelType.KERAS:
            return self.__run_keras_inference(data)
        elif self.type == ModelType.TFLITE:
            return self.__run_tflite_inference(data)
            
        
    def __run_keras_inference(self, data: np.ndarray) -> np.ndarray:
        """Run inference on the input data using a Keras model."""
        return self.model.predict(data, verbose = 0)
    
    def __run_tflite_inference(self, data: np.ndarray) -> np.ndarray:
        """Run inference on the input data using a TensorFlow Lite model."""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        input_type = input_details[0]['dtype']
        if input_type == np.int8:
            scale, zero_point = input_details[0]['quantization']
            data = np.clip((data / scale + zero_point), -128, 127).astype(np.int8)
        elif input_type == np.float16:
            data = data.astype(np.float16)
        self.model.set_tensor(input_details[0]['index'], data)
        self.model.invoke()
        return self.model.get_tensor(output_details[0]['index'])

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's predictions using the specified metrics.

        Args:
            predictions (np.ndarray): Model's predictions.
            labels (np.ndarray): True labels.

        Returns:
            Dict[str, float]: A dictionary of metric names and their corresponding values.
        """
        return {name: metric_func(predictions, labels) for name, metric_func in self.metrics.items()}

    def get_size(self) -> float:
        """
        Get the size of the model file.

        Returns:
            float: Size of the model file in megabytes.
        """
        return os.path.getsize(self.path) / (1024)  # Size in KB

    @staticmethod
    def _default_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the default accuracy metric for classification tasks.

        Args:
            predictions (np.ndarray): Model's predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: The computed accuracy.
        """
        predictions_array = np.array(predictions).squeeze()  # Remove the extra dimension
        predicted_labels = np.argmax(predictions_array, axis=1)
        return np.mean(predicted_labels == labels)