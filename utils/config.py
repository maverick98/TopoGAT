import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, config_path="config/config.yaml"):
    """
    Save a dictionary to a YAML configuration file.

    Args:
        config (dict): Configuration dictionary.
        config_path (str): Path to save the YAML config.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
