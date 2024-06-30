# models/base_model.py

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def benchmark(self):
        """
        This method should implement the benchmarking logic.
        """
        pass

    @abstractmethod
    def convert(self):
        """
        This method should implement the conversion logic.
        """
        pass