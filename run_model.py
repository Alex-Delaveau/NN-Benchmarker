import os
import sys
import importlib

def run_model(model_name, operation):
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")

    # Import the run module from the specified model
    try:
        print(f"Attempting to import: models.{model_name}.run")
        module = importlib.import_module(f"models.{model_name}.run")
    except ImportError as e:
        print(f"Error: Could not import the run module for model '{model_name}'")
        print(f"Import error details: {e}")
        return

    # Check if the module has a get_model function
    if hasattr(module, "get_model"):
        model = module.get_model()
        if operation == "benchmark" or operation == "both":
            model.benchmark()
        if operation == "convert" or operation == "both":
            model.convert()
    else:
        print(f"Error: The run module for model '{model_name}' does not have a get_model function")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_model.py <model_name> <operation>")
        print("Operation can be: 'benchmark', 'convert', or 'both'")
        sys.exit(1)

    model_name = sys.argv[1]
    operation = sys.argv[2].lower()

    if operation not in ["benchmark", "convert", "both"]:
        print("Invalid operation. Please use 'benchmark', 'convert', or 'both'.")
        sys.exit(1)

    run_model(model_name, operation)