import os

def get_data_dir():
    """Get the data directory from environment variable or default."""
    return os.environ.get('DMC_DISCRETE_DATA_DIR')
