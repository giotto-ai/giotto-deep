import os
from os.path import exists
from pathlib import Path
from zenodo_client.api import Zenodo

import yaml

from gdeep.data.datasets.cloud.utils import get_config_path, get_download_directory
from gdeep.utility.constants import DEFAULT_DOWNLOAD_DIR


class DatasetCloud:
    def __init__(
        self,
        dataset_name: str,
        root_download_directory: str=DEFAULT_DOWNLOAD_DIR
    ):
        self.name = dataset_name
        self.zenodo = Zenodo()
        self.download_directory = get_download_directory(self.name, root_download_directory)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = get_config_path(self.name)
        if not exists(config_path):
            raise ValueError(f"Configuration file {config_path} does not exist.")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config


    def does_dataset_exist_locally(self) -> bool:
        return exists(self.download_directory / self.name)

    def download(self) -> None:
        os.makedirs(self.download_directory, exist_ok=True)

        for file in self.config["files"]:
            self.zenodo.download(self.config["deposition_id"], file, parts=[str(self.download_directory)])





