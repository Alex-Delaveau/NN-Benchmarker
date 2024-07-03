# TFLite Model Optimization and Benchmarking Application

This application is designed to convert Keras models to TensorFlow Lite (TFLite) models using quantization and to benchmark multiple models.

## App Architecture

### Directory Structure

- **Benchmarks/**:
  Contains all the code used for benchmarking models.

- **Quantization_tools/**:
  Contains the code to perform quantization on models.

- **Models/**:
  Contains the models you want to convert or quantize. Each model directory must follow the implementation examples (e.g., Mnist or CloudNet).
  - Each model directory should include a `run.py` script with a class representing your model that extends `BaseModel`. This class must have `convert` and `benchmark` functions to handle conversion to TFLite and benchmarking, respectively.

### Main Script

- **run_model.py**:
  The main script of the application used to launch the processes.
  - Usage: `python run_model.py {model_name} {action}`
    - `{model_name}`: The name of the model directory (e.g., `mnist`).
    - `{action}`: The action to perform (`benchmark` or `convert`).

## Usage Example

To start the benchmark function for the MNIST model, run:
```bash
python run_model.py mnist benchmark
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Alex-Delaveau/TFLite-Optimize-Bench.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you want to work on the CloudNet model, install its specific dependencies:
   ```bash
   pip install -r models/cloudnet/requirements.txt
   ```

4. Run the desired model and action:
   ```bash
   python run_model.py {model_name} {action}
   ```

Replace `{model_name}` with the name of your model (e.g., `mnist`) and `{action}` with the desired action (`benchmark` or `convert`).
