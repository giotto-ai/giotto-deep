from pathlib import Path
from gdeep.utility.constants import DATASET_CLOUD_CONFIGS_DIR

def get_config_path(dataset_name: str) -> Path:
    return Path(DATASET_CLOUD_CONFIGS_DIR) / f"{dataset_name}.yaml"

def get_download_directory(dataset_name: str, root_download_directory: str) -> Path:
    return Path(root_download_directory) / dataset_name

def get_dataset_list() -> list[str]:
    return [path.stem for path in Path(DATASET_CLOUD_CONFIGS_DIR).iterdir() if path.suffix == ".yaml"]