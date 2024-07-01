# models/mnist/run.py

from benchmarks.model_comparator import ModelComparator
from models.base_model import BaseModel
from benchmarks.benchmark_model import BenchmarkModel, ModelType
from benchmarks.benchmarker import Benchmarker
import tensorflow as tf
import numpy as np

class MNISTModel(BaseModel):
    def convert(self):
        raise NotImplementedError("Conversion not implemented for MNIST model")

    def benchmark(self):
        print("Benchmarking MNIST model...")
        (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        test_images_sample = test_images[:100]
        test_labels_sample = test_labels[:100]

        def preprocess_function(images):
            # Ensure the input is a numpy array
            images = np.array(images)
            
            # Get the shape of the input images
            shape = images.shape
            
            # Check the number of dimensions in the input images
            if len(shape) == 3:
                # If the input images have 3 dimensions, reshape to add one more dimension
                return (images.astype(np.float32) / 255.0)
            elif len(shape) == 4:
                # If the input images already have 4 dimensions, remove the last dimension
                return (np.squeeze(images, axis=-1).astype(np.float32) / 255.0)
            else:
                # Handle other cases if needed
                raise ValueError("Unexpected input shape: {}".format(shape))

        
        keras_model = BenchmarkModel(
            name="KerasModel", 
            model_path='models/mnist/saved_models/mnist_model.keras', 
            model_type=ModelType.KERAS, 
            test_dataset=(test_images_sample, test_labels_sample),
            preprocess_func=preprocess_function
        )
        tflite_model = BenchmarkModel(
            name="TFLiteModel", 
            model_path='models/mnist/saved_models/mnist_model8.tflite', 
            model_type=ModelType.TFLITE, 
            test_dataset=(test_images_sample, test_labels_sample),
            preprocess_func=preprocess_function
        )
        
        benchmarker = Benchmarker()
        benchmarker.add_model(keras_model)
        benchmarker.add_model(tflite_model)
        results = benchmarker.compare_all_models()
        comparator = ModelComparator(results)
        comparator.print_comparison()

def get_model():
    return MNISTModel()