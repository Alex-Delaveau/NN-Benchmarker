from typing import Tuple, Callable, Optional
from enum import Enum
import os
import numpy as np
import tensorflow as tf

class ModelType(Enum):
    KERAS = "keras"
    TFLITE = "tflite"

class BenchmarkModel:
    def __init__(self, name: str, model_path: str, model_type: ModelType, 
                 dataset: Tuple[np.ndarray, np.ndarray], 
                 preprocess_func: Callable[[np.ndarray], np.ndarray] = None,
                 metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None):
        self.name = name
        self.path = model_path
        self.type = model_type
        self.X, self.y = dataset
        self.preprocess_func = preprocess_func or (lambda x: x)
        self.metric_func = metric_func
        self._load_model()

    def _load_model(self):
        if self.type == ModelType.KERAS:
            self.model = tf.keras.models.load_model(self.path)
        elif self.type == ModelType.TFLITE:
            self.model = tf.lite.Interpreter(model_path=self.path)
            self.model.allocate_tensors()
        else:
            raise ValueError(f"Unsupported model type: {self.type}")

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return self.preprocess_func(data)

    def run_inference(self, data: np.ndarray) -> np.ndarray:
        if self.type == ModelType.KERAS:
            return self.model.predict(data)
        elif self.type == ModelType.TFLITE:
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

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        if self.metric_func:
            return self.metric_func(predictions, labels)
        else:
            raise NotImplementedError("No metric function provided for model evaluation")


    def get_size(self) -> float:
        return os.path.getsize(self.path) / (1024 * 1024)  # Size in MB
