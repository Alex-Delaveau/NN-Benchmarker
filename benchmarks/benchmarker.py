"""
This module provides a Benchmarker class for comparing multiple machine learning models.

The Benchmarker class allows for adding multiple BenchmarkModel instances,
measuring their inference time and metrics, and comparing their performance.

Example usage:
    from benchmarker import Benchmarker
    from benchmark_model import BenchmarkModel, ModelType
    import numpy as np

    # Prepare your data
    X = np.random.rand(1000, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 10, 1000)

    # Create BenchmarkModel instances
    keras_model = BenchmarkModel("KerasModel", 'path/to/keras_model.h5', ModelType.KERAS, (X, y))
    tflite_model = BenchmarkModel("TFLiteModel", 'path/to/tflite_model.tflite', ModelType.TFLITE, (X, y))

    # Create Benchmarker and add models
    benchmarker = Benchmarker()
    benchmarker.add_model(keras_model)
    benchmarker.add_model(tflite_model)

    # Run comparison
    results = benchmarker.compare_all_models()
    results = benchmarker.compare_models(["KerasModel", "TFLiteModel"])

    # Print results
    benchmarker.print_comparison(results)
"""

from .benchmark_model import BenchmarkModel
import time
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os
import tensorflow as tf
import logging
from tabulate import tabulate

class Benchmarker:
    """
    A class for benchmarking multiple machine learning models.

    This class allows for adding multiple BenchmarkModel instances,
    measuring their inference time and metrics, and comparing their performance.
    """

    def __init__(self):
        """Initialize a Benchmarker instance with an empty dictionary of models."""
        self.models = {}

    def set_tf_logging(self, level):
        """
        Set TensorFlow logging level.

        Args:
        level (str or int): Desired logging level. Can be 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', or 0-4.
        """
        tf_logging_levels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 'DEBUG': '0', 'INFO': '1', 'WARN': '2', 'ERROR': '3', 'FATAL': '4'}
        python_logging_levels = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN, 3: logging.ERROR, 4: logging.FATAL, 'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARN': logging.WARN, 'ERROR': logging.ERROR, 'FATAL': logging.FATAL}

        if isinstance(level, int) and level in tf_logging_levels:
            os_level = tf_logging_levels[level]
            py_level = python_logging_levels[level]
        elif isinstance(level, str) and level.upper() in tf_logging_levels:
            os_level = tf_logging_levels[level.upper()]
            py_level = python_logging_levels[level.upper()]
        else:
            raise ValueError("Invalid logging level. Choose from 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', or 0-4")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = os_level
        tf.get_logger().setLevel(py_level)

    def add_model(self, model: BenchmarkModel):
        """
        Add a BenchmarkModel instance to the Benchmarker.

        Args:
            model (BenchmarkModel): The model to be added.
        """
        self.models[model.name] = model

    def measure_inference_time(self, model_name: str, num_runs: int = 100) -> float:
        """
        Measure the average inference time for a specific model.

        Args:
            model_name (str): The name of the model to measure.
            num_runs (int, optional): The number of inference runs. Defaults to 100.

        Returns:
            float: The average inference time in milliseconds.
        """
        print("Measuring inference time for model:", model_name)
        model = self.models[model_name]
        data = model.preprocess_data(model.X_test[:num_runs])
        
        start_time = time.time()
        for sample in data:
            model.run_inference(np.expand_dims(sample, axis=0))
        print("Finished inference for model:", model_name)
        duration = time.time() - start_time
        return (duration / num_runs) * 1000  # Convert to milliseconds

    def measure_metrics(self, model_name: str) -> Dict[str, float]:
        """
        Measure the performance metrics for a specific model.

        Args:
            model_name (str): The name of the model to measure.

        Returns:
            Dict[str, float]: A dictionary of metric names and their corresponding values.
        """
        print("Measuring metrics for model:", model_name)
        
        predictions = []
        model = self.models[model_name]
        data = model.preprocess_data(model.X_test)
        for sample in tqdm(data, desc="Processing Samples"):
            predictions.append(model.run_inference(np.expand_dims(sample, axis=0)))
        print("Finished metrics for model:", model_name)
        return model.evaluate(predictions, model.y_test)

    def compare_models(self, model_names: List[str], num_runs: int = 100) -> dict:
        """
        Compare multiple models based on inference time, metrics, and size.

        Args:
            model_names (List[str]): A list of model names to compare.
            num_runs (int, optional): The number of inference runs for time measurement. Defaults to 100.

        Returns:
            dict: A dictionary containing comparison results for each model.
        """
        results = {}
        current_logging_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
        self.set_tf_logging(3) 
        for name in model_names:
            print(f"Comparing model: {name}")
            model = self.models[name]
            results[name] = {
                "inference_time": self.measure_inference_time(name, num_runs),
                "metrics": self.measure_metrics(name),
                "size": model.get_size()
            }
            
        self.set_tf_logging(int(current_logging_level))
        return results
    
    def compare_all_models(self, num_runs: int = 100) -> dict:
        """
        Compare all models added to the Benchmarker based on inference time, metrics, and size.

        Args:
            num_runs (int, optional): The number of inference runs for time measurement. Defaults to 100.

        Returns:
            dict: A dictionary containing comparison results for each model.
        """
        return self.compare_models(list(self.models.keys()), num_runs)
