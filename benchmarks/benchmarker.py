from benchmark_model import BenchmarkModel
import time
import numpy as np
from typing import List, Optional, Callable


class Benchmarker:
    def __init__(self):
        self.models = {}

    def add_model(self, model: BenchmarkModel):
        self.models[model.name] = model

    def measure_inference_time(self, model_name: str, num_runs: int = 100) -> float:
        model = self.models[model_name]
        data = model.preprocess_data(model.X[:num_runs])

        start_time = time.time()
        for sample in data:
            model.run_inference(np.expand_dims(sample, axis=0))
        duration = time.time() - start_time
        return (duration / num_runs) * 1000  # Convert to milliseconds

    def measure_accuracy(self, model_name: str) -> Optional[float]:
        model = self.models[model_name]
        data = model.preprocess_data(model.X)
        predictions = model.run_inference(data)
        return model.evaluate(predictions, model.y)

    def compare_models(self, model_names: List[str], num_runs: int = 100) -> dict:
        results = {}
        for name in model_names:
            model = self.models[name]
            results[name] = {
                "inference_time": self.measure_inference_time(name, num_runs),
                "accuracy": self.measure_accuracy(name),
                "size": model.get_size()
            }
        return results

    def print_comparison(self, results: dict):
        print(f"{'Model':<20} {'Inference Time (ms)':<20} {'Accuracy':<20} {'Size (MB)':<20}")
        print("-" * 80)
        for model, metrics in results.items():
            print(f"{model:<20} {metrics['inference_time']:<20.2f} {metrics['accuracy']:<20.2f} {metrics['size']:<20.2f}")