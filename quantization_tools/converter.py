import os
import tensorflow as tf
from enum import Enum, auto
from typing import Callable, Optional

class QuantizationType(Enum):
    """
    Enumeration of quantization types supported for model conversion.
    """
    INT8 = auto()   # 8-bit integer quantization
    FP16 = auto()   # 16-bit floating point quantization
    INT16 = auto()  # 16-bit integer quantization

class Converter:
    """
    A class to convert a TensorFlow Keras model to TensorFlow Lite format with quantization.

    Attributes:
        model (tf.keras.Model): The Keras model to be converted.
        converter (tf.lite.TFLiteConverter): Internal converter used to perform the conversion.
        tflite_model (bytes): The converted TensorFlow Lite model.
    Example:
        >>> import tensorflow as tf
        >>> # Load or define your Keras model
        >>> model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
        >>> # Initialize the converter with the Keras model
        >>> converter = Converter(model)
        >>> # Define a representative dataset generator (required for some quantization types)
        >>> def representative_dataset_gen():
        ...     for _ in range(100):
        ...         # Assuming a batch size of 1
        ...         yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]
        >>> # Create, convert, and save the model with INT8 quantization
        >>> converter.create_convert_and_save(QuantizationType.INT8, 'saved_models', 'converted_model.tflite', representative_dataset_gen)
        >>> # Get the size of the saved model
        >>> size = Converter.get_model_size('saved_models/converted_model.tflite')
        >>> print(f"Model size: {size:.2f} MB")
    """
    def __init__(self, model):
        """
        Initializes the Converter with a Keras model.

        Parameters:
            model (tf.keras.Model): TensorFlow Keras model to be converted.
        """
        self.model = model
        self.converter = None
        self.tflite_model = None

    def create_converter(self, quant_type: QuantizationType, representative_data_gen: Optional[Callable] = None):
        """
        Creates a converter configured for the specified quantization type.

        Parameters:
            quant_type (QuantizationType): The type of quantization to apply.
            representative_data_gen (Callable, optional): A function that yields batches of input data for calibration.
        """
        self.converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quant_type == QuantizationType.INT8:
            self._setup_int8_converter(representative_data_gen)
        elif quant_type == QuantizationType.FP16:
            self._setup_fp16_converter()
        elif quant_type == QuantizationType.INT16:
            self._setup_int16_converter(representative_data_gen)
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")

    def _setup_int8_converter(self, representative_data_gen: Callable):
        """
        Sets up the converter for INT8 quantization.

        Parameters:
            representative_data_gen (Callable): Function providing representative data needed for quantization.
        """
        if representative_data_gen is None:
            raise ValueError("Representative data generator is required for INT8 quantization")
        self.converter.representative_dataset = representative_data_gen
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8

    def _setup_fp16_converter(self):
        """
        Sets up the converter for FP16 quantization.
        """
        self.converter.target_spec.supported_types = [tf.float16]

    def _setup_int16_converter(self, representative_data_gen: Callable):
        """
        Sets up the converter for INT16 quantization.

        Parameters:
            representative_data_gen (Callable): Function providing representative data needed for quantization.
        """
        if representative_data_gen is None:
            raise ValueError("Representative data generator is required for INT16 quantization")
        self.converter.representative_dataset = representative_data_gen
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

    def convert(self):
        """
        Converts the model using the previously created converter.

        Returns:
            bytes: The converted TensorFlow Lite model.
        """
        if self.converter is None:
            raise ValueError("Converter not created. Call create_converter() first.")
        self.tflite_model = self.converter.convert()
        return self.tflite_model

    def save(self, save_dir: str, model_name: str):
        """
        Saves the converted TensorFlow Lite model to a specified directory.

        Parameters:
            save_dir (str): The directory to save the converted model.
            model_name (str): The filename for the saved model.
        """
        if self.tflite_model is None:
            raise ValueError("No converted model to save. Call convert() first.")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_name)
        with open(save_path, "wb") as f:
            f.write(self.tflite_model)

    def create_convert_and_save(self, quant_type: QuantizationType, save_dir: str, model_name: str, representative_data_gen: Optional[Callable] = None):
        """
        A comprehensive method that creates the converter, converts the model, and saves the converted model.

        Parameters:
            quant_type (QuantizationType): The type of quantization to apply.
            save_dir (str): Directory to save the converted model.
            model_name (str): Filename for the saved model.
            representative_data_gen (Callable, optional): Function providing representative data for quantization.
        """
        self.create_converter(quant_type, representative_data_gen)
        self.convert()
        self.save(save_dir, model_name)

    @staticmethod
    def get_model_size(file_path: str) -> float:
        """
        Returns the size of a file in megabytes.

        Parameters:
            file_path (str): The path to the file.

        Returns:
            float: The size of the file in megabytes.
        """
        return os.path.getsize(file_path) / 1024  # Size in KB