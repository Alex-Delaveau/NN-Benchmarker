from models.base_model import BaseModel

class CloudNetModel(BaseModel):
    def convert(self):
        raise NotImplementedError("Conversion not implemented for CloudNet model")

    def benchmark(self):
        raise NotImplementedError("Conversion not implemented for CloudNet model")
    


def get_model():
    return CloudNetModel()