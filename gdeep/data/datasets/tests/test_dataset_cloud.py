import os
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import yaml
from zenodo_client.struct import Metadata
from gdeep.data.datasets.cloud.dataset_cloud import DatasetCloud
from gdeep.data.datasets.cloud.dataset_uploader import DatasetUploader
from gdeep.data.datasets.cloud.utils import get_config_path, get_dataset_list

def test_dataset_upload_and_download():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data and metadata
        data_tensor = torch.randn(10, 5)
        labels_tensor = torch.randint(0, 2, (10,))
        metadata = Metadata(title="Test Dataset", description="A test dataset for unit testing")

        # Save data and metadata to temporary files
        data_path = Path(temp_dir) / "data.pt"
        labels_path = Path(temp_dir) / "labels.pt"
        metadata_path = Path(temp_dir) / "metadata.yaml"
        torch.save(data_tensor, data_path)
        torch.save(labels_tensor, labels_path)
        with open(metadata_path, "w") as f:
            yaml.dump(dict(metadata), f)

        # Upload the dataset
        dataset_name = "test_dataset"
        file_paths = [str(data_path), str(labels_path), str(metadata_path)]
        uploader = DatasetUploader(sandbox=True)
        uploader.upload(dataset_name, metadata, file_paths)

        try:
            # Check if the dataset is added to the list of datasets
            assert dataset_name in get_dataset_list()

            # Check if the dataset config file is created
            config_path = get_config_path(dataset_name)
            assert config_path.exists()

            # Download the dataset
            download_dir = Path(temp_dir) / "download"
            dataset = DatasetCloud(dataset_name, root_download_directory=str(download_dir))
            dataset.download()

            # Check if the downloaded files exist
            downloaded_data_path = download_dir / dataset_name / "data.pt"
            downloaded_labels_path = download_dir / dataset_name / "labels.pt"
            assert downloaded_data_path.exists()
            assert downloaded_labels_path.exists()

            # Check if the downloaded data matches the original data
            downloaded_data = torch.load(downloaded_data_path)
            downloaded_labels = torch.load(downloaded_labels_path)
            assert torch.allclose(downloaded_data, data_tensor)
            assert torch.allclose(downloaded_labels, labels_tensor)

        finally:
            # Clean up the uploaded dataset
            uploader.remove(dataset_name)
            assert not config_path.exists()
            assert dataset_name not in get_dataset_list()

def test_dataset_update():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data and metadata
        data_tensor = torch.randn(10, 5)
        labels_tensor = torch.randint(0, 2, (10,))
        metadata = Metadata(title="Test Dataset", description="A test dataset for unit testing")

        # Save data and metadata to temporary files
        data_path = Path(temp_dir) / "data.pt"
        labels_path = Path(temp_dir) / "labels.pt"
        metadata_path = Path(temp_dir) / "metadata.yaml"
        torch.save(data_tensor, data_path)
        torch.save(labels_tensor, labels_path)
        with open(metadata_path, "w") as f:
            yaml.dump(dict(metadata), f)

        # Upload the dataset
        dataset_name = "test_dataset"
        file_paths = [str(data_path), str(labels_path), str(metadata_path)]
        uploader = DatasetUploader(sandbox=True)
        uploader.upload(dataset_name, metadata, file_paths)

        try:
            # Update the dataset with new files
            updated_data_tensor = torch.randn(20, 5)
            updated_labels_tensor = torch.randint(0, 2, (20,))
            updated_data_path = Path(temp_dir) / "updated_data.pt"
            updated_labels_path = Path(temp_dir) / "updated_labels.pt"
            torch.save(updated_data_tensor, updated_data_path)
            torch.save(updated_labels_tensor, updated_labels_path)
            updated_file_paths = [str(updated_data_path), str(updated_labels_path)]
            uploader.update(dataset_name, updated_file_paths)

            # Download the updated dataset
            download_dir = Path(temp_dir) / "download"
            dataset = DatasetCloud(dataset_name, root_download_directory=str(download_dir))
            dataset.download()

            # Check if the downloaded files exist
            downloaded_data_path = download_dir / dataset_name / "updated_data.pt"
            downloaded_labels_path = download_dir / dataset_name / "updated_labels.pt"
            assert downloaded_data_path.exists()
            assert downloaded_labels_path.exists()

            # Check if the downloaded data matches the updated data
            downloaded_data = torch.load(downloaded_data_path)
            downloaded_labels = torch.load(downloaded_labels_path)
            assert torch.allclose(downloaded_data, updated_data_tensor)
            assert torch.allclose(downloaded_labels, updated_labels_tensor)

        finally:
            # Clean up the uploaded dataset
            uploader.remove(dataset_name)

def test_dataset_remove():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data and metadata
        data_tensor = torch.randn(10, 5)
        labels_tensor = torch.randint(0, 2, (10,))
        metadata = Metadata(title="Test Dataset", description="A test dataset for unit testing")

        # Save data and metadata to temporary files
        data_path = Path(temp_dir) / "data.pt"
        labels_path = Path(temp_dir) / "labels.pt"
        metadata_path = Path(temp_dir) / "metadata.yaml"
        torch.save(data_tensor, data_path)
        torch.save(labels_tensor, labels_path)
        with open(metadata_path, "w") as f:
            yaml.dump(dict(metadata), f)

        # Upload the dataset
        dataset_name = "test_dataset"
        file_paths = [str(data_path), str(labels_path), str(metadata_path)]
        uploader = DatasetUploader(sandbox=True)
        uploader.upload(dataset_name, metadata, file_paths)

        # Remove the dataset
        uploader.remove(dataset_name)

        # Check if the dataset is removed from the list of datasets
        assert dataset_name not in get_dataset_list()

        # Check if the dataset config file is removed
        config_path = get_config_path(dataset_name)
        assert not config_path.exists()


def test_all_datasets_valid():
    with tempfile.TemporaryDirectory() as temp_dir:
        download_dir = Path(temp_dir) / "download"

        # Get the list of all datasets in DATASET_CLOUD_CONFIGS_DIR
        dataset_list = get_dataset_list()

        # Download each dataset and check for errors
        for dataset_name in dataset_list:
            dataset = DatasetCloud(dataset_name, root_download_directory=str(download_dir))

            try:
                dataset.download()

                # Check if the downloaded files exist
                downloaded_files = list(Path(download_dir / dataset_name).glob("*"))
                dataset_files = dataset.config["files"]
                assert len(downloaded_files) == len(dataset_files), \
                    f"Number of files do not match for dataset: {dataset_name}"

            except Exception as e:
                pytest.fail(f"Error occurred while downloading dataset: {dataset_name}\n{str(e)}")

            finally:
                # Clean up the downloaded dataset
                shutil.rmtree(download_dir / dataset_name)