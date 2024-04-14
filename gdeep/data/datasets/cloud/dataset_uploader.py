from typing import Optional
import os
import yaml
from pathlib import Path
from zenodo_client.api import Zenodo
from zenodo_client.struct import Metadata
from gdeep.data.datasets.cloud.utils import get_config_path

class DatasetUploader:
    def __init__(self, access_token: Optional[str] = None, sandbox: bool = False):
        if access_token is None:
            access_token = os.getenv("ZENODO_API_TOKEN")
        self.zenodo_client = Zenodo(access_token=access_token, sandbox=sandbox)

    def upload(self, dataset_name: str, metadata: Metadata, file_paths: list[str]) -> None:
        config_path = get_config_path(dataset_name)
        if os.path.exists(config_path):
            raise FileExistsError(f"Configuration file already exists for dataset: {dataset_name}, use update method instead")
        deposition = self.zenodo_client.create(data=metadata, paths=file_paths)
        deposition_id = deposition["id"]
        file_names = [file["filename"] for file in deposition["files"]]
        config = {
            "deposition_id": deposition_id,
            "files": file_names
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def update(self, dataset_name: str, file_paths: list[str]) -> None:
        config_path = get_config_path(dataset_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found for dataset: {dataset_name}")
        with open(get_config_path(dataset_name), "r") as f:
            config = yaml.safe_load(f)
        deposition_id = config["deposition_id"]
        deposition = self.zenodo_client.update(deposition_id, file_paths)
        config["files"] = [file["filename"] for file in deposition["files"]]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def remove(self, dataset_name: str) -> None:
        config_path = get_config_path(dataset_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found for dataset: {dataset_name}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        deposition_id = config["deposition_id"]
        self.zenodo_client.discard(deposition_id)
        os.remove(config_path)

