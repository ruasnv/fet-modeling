# scripts/run_full_pipeline.py
import shutil
import subprocess
from pathlib import Path
import sys
def clean_directories(base_path):
    """
    Deletes and recreates specified directories to ensure a clean slate.

    Args:
        base_path (Path): The base path of the project
    """
    dirs_to_clean = [
        base_path / "results",
        base_path / "models",
        base_path / "data" / "processed"
    ]

    print("\n    Cleaning previous output directories    ")
    for d in dirs_to_clean:
        if d.exists():
            print(f"Deleting: {d}")
            shutil.rmtree(d)
        print(f"Creating: {d}")
        d.mkdir(parents=True, exist_ok=True) # Recreate, ensuring parent dirs exist
    print("   Cleaning complete    ")

def run_script(script_path):
    """
    Runs a Python script using subprocess, ensuring the correct Python interpreter is used.

    Args:
        script_path (Path): The full path to the Python script to run.
    """
    print(f"\n   Running {script_path.name}    ")
    try:
        # Use sys.executable to ensure the interpreter from the virtual environment is used
        result = subprocess.run([sys.executable, str(script_path)], check=True, capture_output=False)
        print(f"    {script_path.name} finished successfully    ")
    except subprocess.CalledProcessError as e:
        print(f"    Error running {script_path.name}: {e}    ")
        print(f"  Stdout: {e.stdout.decode() if e.stdout else 'None'}")
        print(f"  Stderr: {e.stderr.decode() if e.stderr else 'None'}")
        # Exit if any script fails
        exit(1)
    except FileNotFoundError:
        print(f"    Error: Python executable not found. Make sure Python is in your PATH or virtual environment is active.    ")
        exit(1)

def main():
    """
    Orchestrates the full MOSFET modeling pipeline.
    """
    # Determine the project's base directory dynamically
    project_root = Path(__file__).resolve().parent.parent

    # Define the paths to the scripts to run
    scripts_dir = project_root / "scripts"
    pipeline_scripts = [
        scripts_dir / "run_data_processing.py",
        scripts_dir / "run_eda.py",
        #scripts_dir / "run_gan_augmentation.py",
        scripts_dir / "run_training_simple_nn.py",
        scripts_dir / "run_evaluation_on_model.py"
    ]

    #Clean previous output directories
    clean_directories(project_root)

    #Run each script in sequence
    for script in pipeline_scripts:
        if not script.exists():
            print(f"Error: Script not found: {script}. Please ensure all pipeline scripts exist.")
            exit(1)
        run_script(script)

    print("\n   Full FET Modeling Pipeline Execution Complete   ")

if __name__ == "__main__":
    main()