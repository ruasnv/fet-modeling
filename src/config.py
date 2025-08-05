# src/config.py
import yaml
import os
from pathlib import Path


class AppConfig:
    _instance = None
    _config_data = {}  # This will store all your merged config variables
    _project_root = None  # Store the project root once found

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _find_project_root(self, start_path):
        """
        Finds the project root by looking for a 'config' directory and a .git folder or pyproject.toml
        or README.md as a marker.
        """
        p = Path(start_path).resolve()
        while p != p.parent:
            # Look for common project markers in addition to 'config' directory
            if (p / "config").is_dir() and \
                    ((p / ".git").is_dir() or (p / "pyproject.toml").is_file() or (p / "README.md").is_file()):
                return p
            p = p.parent
        raise FileNotFoundError(
            "Project root not found. Please ensure a 'config' directory "
            "and a project marker (e.g., .git, pyproject.toml, README.md) exist in a parent directory."
        )

    def _load_config(self):
        try:
            this_file_dir = Path(__file__).parent
            self._project_root = self._find_project_root(this_file_dir)
            #print(f"DEBUG: Project root found at: {self._project_root}")

            config_dir = self._project_root / "config"
            if not config_dir.is_dir():
                raise FileNotFoundError(f"Config directory not found at {config_dir}")

            config_files = sorted(list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml")))
            if not config_files:
                raise FileNotFoundError(f"No YAML config files found in {config_dir}")

            # Merge configurations. Order matters for overrides:
            # If main_config.yaml, data_config.yaml, simple_nn_config.yaml are loaded in that order,
            # simple_nn_config.yaml can override keys from data_config.yaml.
            # `sorted` ensures consistent loading order if files have similar names.
            for config_file_path in config_files:
                #print(f"DEBUG: Loading config file: {config_file_path}")
                with open(config_file_path, "rt") as f:
                    loaded_data = yaml.safe_load(f)
                    if loaded_data:
                        self._config_data.update(loaded_data)  # Merges, overriding duplicates

            # Define the specific keys within your merged config that represent paths
            # These are the *final* keys that will hold Path objects after processing
            # Ensure these match the structure of your merged config data
            path_keys_to_process = [
                "paths.raw_data_path",
                "paths.processed_data_dir",
                "paths.trained_model_dir",
                "paths.report_output_dir",
                "paths.eda_output_dir",
                "paths.plots_output_dir",
                # Add any other keys from your config that represent relative paths
                # e.g., if you had 'data.specific_file_path'
            ]

            # Process all relative paths found in _config_data to be absolute Path objects
            for full_key_path in path_keys_to_process:
                parts = full_key_path.split('.')
                current_dict = self._config_data
                original_value = None  # Store the actual string value
                target_key = parts[-1]  # The last part is the key we want to modify

                # Traverse to the dictionary containing the actual path string
                found_container = True
                for part in parts[:-1]:  # Iterate up to the second to last part
                    if isinstance(current_dict, dict) and part in current_dict:
                        current_dict = current_dict[part]
                    else:
                        found_container = False
                        break

                if found_container and isinstance(current_dict, dict) and target_key in current_dict:
                    original_value = current_dict[target_key]
                    if isinstance(original_value, str):
                        current_dict[target_key] = self._project_root / original_value
                        #print(f"DEBUG: Processed path '{full_key_path}': {current_dict[target_key]}")
                    else:
                        print(
                            f"WARNING: Expected string for path '{full_key_path}', but got '{type(original_value)}'. Skipping path processing.")
                elif found_container:  # target_key not in current_dict
                    print(f"WARNING: Key '{full_key_path}' not found in configuration for path processing.")
                else:  # Container not found
                    print(f"WARNING: Path '{full_key_path}' could not be fully traversed for path processing.")

            # Ensure output directories exist after path processing
            if "paths" in self._config_data and isinstance(self._config_data["paths"], dict):
                for dir_key in ["processed_data_dir", "trained_model_dir", "report_output_dir", "eda_output_dir",
                                "plots_output_dir"]:
                    full_path = self._config_data["paths"].get(dir_key)
                    # Check if it's a Path object (meaning it was successfully processed)
                    if full_path and isinstance(full_path, Path):
                        os.makedirs(full_path, exist_ok=True)
                        #print(f"DEBUG: Ensured directory exists: {full_path}")
                    elif full_path:
                        print(
                            f"WARNING: Path '{full_path}' for key '{dir_key}' is not a pathlib.Path object. Skipping directory creation.")

        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise  # Re-raise to stop execution if config fails

    # Method to get any config value
    def get(self, key, default=None):
        """Retrieve a configuration value by key, supporting dotted paths."""
        parts = key.split('.')
        current_data = self._config_data
        for part in parts:
            if isinstance(current_data, dict) and part in current_data:
                current_data = current_data[part]
            else:
                return default  # Key or path segment not found
        return current_data

    # Make the entire config dictionary accessible (optional, but convenient for many variables)
    @property
    def all_config(self):
        return self._config_data


# Create a global instance of AppConfig that can be imported
settings = AppConfig()