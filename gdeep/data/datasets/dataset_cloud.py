import os
from os import remove
from os.path import join, exists
from typing import List, Tuple, Union, Set

import json
import wget  # type: ignore

from ._data_cloud import _DataCloud  # type: ignore
from gdeep.utility.constants import DEFAULT_DOWNLOAD_DIR, DATASET_BUCKET_NAME


class DatasetCloud:
    """DatasetCloud class to handle the download and upload
    of datasets to the DataCloud.
    If the download_directory does not exist, it will be created and
    if a folder with the same name as the dataset exists in the
    download directory, it will not be downloaded again.
    If a folder with the same name as the dataset does not exists
    locally, it will be created when downloading the dataset.

    Args:
        dataset_name (str):
            Name of the dataset to be downloaded or uploaded.
        bucket_name (str, optional):
            Name of the bucket in the DataCloud.
            Defaults to DATASET_BUCKET_NAME.
        download_directory (Union[None, str], optional):
            Directory where the
            dataset will be downloaded to. Defaults to DEFAULT_DOWNLOAD_DIR.
        use_public_access (bool, optional):
            If True, the dataset will be
            downloaded via public url. Defaults to False.
        path_credentials (Union[None, str], optional):
            Path to the credentials file.
            Only used if public_access is False and credentials are not
            provided. Defaults to None.
        make_public (bool, optional):
            If True, the dataset will be made public

        Raises:
            ValueError:
                Dataset does not exits in cloud.

        Returns:
            None
    """

    def __init__(
        self,
        dataset_name: str,
        bucket_name: str = DATASET_BUCKET_NAME,
        download_directory: Union[None, str] = None,
        use_public_access: bool = True,
        path_to_credentials: Union[None, str] = None,
        make_public: bool = True,
    ) -> None:
        # Non-public datasets start with "private_"
        if make_public or use_public_access or dataset_name.startswith("private_"):
            self.name = dataset_name
        else:
            self.name = "private_" + dataset_name
        self.path_metadata = None
        self.use_public_access = use_public_access
        if download_directory is None:
            # If download_directory is None, the dataset will be downloaded
            # to the default directory.
            self.download_directory = DEFAULT_DOWNLOAD_DIR
        else:
            self.download_directory = download_directory

        self._data_cloud = _DataCloud(
            bucket_name=bucket_name,
            download_directory=self.download_directory,
            use_public_access=use_public_access,
            path_to_credentials=path_to_credentials,
        )
        if use_public_access:
            self.public_url = "https://storage.googleapis.com/" + bucket_name + "/"
        self.make_public = make_public

    def __del__(self) -> None:
        """This function deletes the metadata file if it exists.

        Returns:
            None
        """
        if self.path_metadata != None:
            remove(self.path_metadata)  # type: ignore
        return None

    def download(self) -> None:
        """Download a dataset from the DataCloud. If the dataset does not
        exist in the cloud, an exception will be raised. If the dataset
        exists locally in the download directory, the dataset will not be
        downloaded again.

        Raises:
            ValueError:
                Dataset does not exits in cloud.
            ValueError:
                Dataset exists locally but checksums do not match.
        """
        if self.use_public_access:
            self._download_using_url()
        else:
            self._download_using_api()

    def _download_using_api(self) -> None:
        """Downloads the dataset using the DataCloud API.
        If the dataset does not exist in the bucket, an exception will
        be raised. If the dataset exists locally in the download directory,
        the dataset will not be downloaded again.

        Raises:
            ValueError:
                Dataset does not exits in cloud.

        Returns:
            None
        """
        self._check_public_access()
        # List of existing datasets in the cloud.
        existing_datasets: Set[str] = set(
            [
                blob.name.split("/")[0]  # type: ignore
                for blob in self._data_cloud.bucket.list_blobs()  # type: ignore  # type: ignore
                if blob.name != "giotto-deep-big.png"
            ]
        )  # type: ignore
        if self.name not in existing_datasets:
            raise ValueError(
                "Dataset {} does not exist in the cloud.".format(self.name)
                + "Available datasets are: {existing_datasets}."
            )
        if not self._does_dataset_exist_locally():
            self._create_dataset_folder()
        self._data_cloud.download_folder(self.name + "/")

    def _does_dataset_exist_locally(self) -> bool:
        """Check if the dataset exists locally.

        Returns:
            bool: True if the dataset exists locally, False otherwise.
        """
        return exists(join(self.download_directory, self.name))

    def _create_dataset_folder(self) -> None:
        """Creates a folder with the dataset name in the download directory.

        Returns:
            None
        """
        if not exists(join(self.download_directory, self.name)):
            os.makedirs(join(self.download_directory, self.name))

    def _download_using_url(self) -> None:
        """Download the dataset using the public url.
        If the dataset does not exist in the bucket, an exception will
        be raised. If the dataset exists locally in the download directory,
        the dataset will not be downloaded again.

        Raises:
            ValueError:
                Dataset does not exits in cloud.

        Returns:
            None
        """
        # List of existing datasets in the cloud.
        existing_datasets = self.get_existing_datasets()

        # Check if requested dataset exists in the cloud.
        assert (
            self.name in existing_datasets
        ), "Dataset {} does not exist in the cloud.".format(
            self.name
        ) + "Available datasets are: {}.".format(
            existing_datasets
        )

        # If the dataset does not exist locally, create the dataset folder.
        if not self._does_dataset_exist_locally():
            self._create_dataset_folder()

        # Download the dataset (metadata.json, data.pt, labels.pt)
        # by using the public URL.
        self._data_cloud.download_file(self.name + "/metadata.json")
        # load the metadata.json file to get the filetype
        with open(
            join(self.download_directory, self.name, "metadata.json")  # type: ignore
        ) as f:
            metadata = json.load(f)
        # filetype: Literal['pt', 'npy']
        if metadata["data_format"] == "pytorch_tensor":
            filetype = "pt"
        elif metadata["data_format"] == "numpy_array":
            filetype = "npy"
        else:
            raise ValueError(f"Unknown data format: {metadata['data_format']}")
        self._data_cloud.download_file(self.name + "/data." + filetype)
        self._data_cloud.download_file(self.name + "/labels." + filetype)

    def get_existing_datasets(self) -> List[str]:
        """Returns a list of datasets in the cloud.

        Returns:
            List[str]:
                List of datasets in the cloud.
        """
        if self.use_public_access:
            datasets_local = "tmp_datasets.json"
            # Download the dataset list json file using the public URL.
            wget.download(self.public_url + "datasets.json", datasets_local)  # type: ignore
            datasets = json.load(open(datasets_local))

            # Remove duplicates. This has to be fixed in the future.
            datasets = list(set(datasets))

            # Remove the temporary file.
            remove(datasets_local)

            return datasets
        else:
            existing_datasets = [
                blob_name.split("/")[0]
                for blob_name in self._data_cloud.list_blobs()
                if blob_name != "giotto-deep-big.png" and blob_name != "datasets.json"
            ]
            # Remove duplicates.
            existing_datasets = list(set(existing_datasets))

            # Remove dataset that are not public, i.e. start with "private_".
            existing_datasets = [
                dataset
                for dataset in existing_datasets
                if not dataset.startswith("private_")
            ]

            return existing_datasets

    def _update_dataset_list(self) -> None:
        """Updates the dataset list in the datasets.json file.

        Returns:
            None
        """
        self._check_public_access()

        # List of existing datasets in the cloud.
        existing_datasets = self.get_existing_datasets()

        # Save existing datasets to a json file.
        json_file = "tmp_datasets.json"
        json.dump(existing_datasets, open(json_file, "w"))

        # Upload the json file to the cloud.
        self._data_cloud.upload_file(
            json_file,
            "datasets.json",
            make_public=True,
            overwrite=True,
        )

        # Remove the temporary file.
        remove(json_file)

    @staticmethod
    def _get_filetype(path: str) -> str:
        """Returns the file extension from a given path.

        Args:
            path:
                A string path.

        Returns:
            str:
                The file extension.

        Raises:
            None.
        """
        return path.split(".")[-1]

    def _check_public_access(self) -> None:
        """Check if use_public_access is set to False."""
        assert (
            self.use_public_access is False
        ), "Only download functionality is supported for public access."

    def _upload_data(
        self,
        path: str,
    ) -> None:
        """Uploads the data file to a Cloud Storage bucket.

        Args:
        path:
            The path to the data file.

        Raises:
        ValueError:
            If the file type is not supported.

        Returns:
            None
        """
        self._check_public_access()

        filetype = DatasetCloud._get_filetype(path)

        # Check if the file type is supported
        if filetype in ["pt", "npy"]:
            self._data_cloud.upload_file(
                path,
                (self.metadata["name"] + "/data." + filetype),  # type: ignore
                make_public=self.make_public,
                overwrite=False,
            )
        else:
            raise ValueError("File type {} is not supported.".format(filetype))

    def _upload_label(
        self,
        path: str,
    ) -> None:
        """Uploads a set of labels to a remote dataset.

        Args:
            path:
                the path to the labels file.

        Raises:
            ValueError:
                If the file type is not supported.

        Returns:
            None

        """
        self._check_public_access()

        filetype = DatasetCloud._get_filetype(path)

        # Check if the file type is supported
        if filetype in ["pt", "npy"]:
            self._data_cloud.upload_file(
                path,
                (self.metadata["name"] + "/labels." + filetype),  # type: ignore
                make_public=self.make_public,
                overwrite=False,
            )
        else:
            raise ValueError("File type {} is not supported.".format(filetype))

    def _upload_metadata(self, path: Union[str, None] = None) -> None:
        """Uploads the metadata dictionary to the location specified in the
        metadata. The metadata dictionary is generated using create_metadata.

        Args:
            path (str):
                The path to the data cloud folder. If none, path will
                be set to the default path.

        Raises:
            Exception:
                If no metadata exists, an exception will be raised.

        Returns:
            None
        """
        self._check_public_access()
        self._data_cloud.upload_file(
            path,  # type: ignore
            str(self.metadata["name"]) + "/" + "metadata.json",  # type: ignore
            make_public=self.make_public,  # type: ignore
        )

    def _add_metadata(
        self,
        size_dataset: int,
        input_size: Tuple[int, ...],
        num_labels: Union[None, int] = None,
        data_type: str = "tabular",
        task_type: str = "classification",
        name: Union[None, str] = None,
        data_format: Union[None, str] = None,
        comment: Union[None, str] = None,
    ) -> None:
        """This function accepts various metadata for the dataset and stores it
        in a temporary JSON file.

        Args:
            size_dataset (int):
                The size of the dataset (in terms of the number
                of samples).
            input_size (Tuple[int, ...]):
                The size of each sample in the
                dataset.
            num_labels (Union[None, int]):
                The number of classes in the dataset.
            data_type (str):
                The type of data in the dataset.
            task_type (str):
                The task type of the dataset.
            name (Union[None, str]):
                The name of the dataset.
            data_format (Union[None, str]):
                The format of the data in the dataset.
            comment (Union[None, str]):
                A comment describing the dataset.

        Returns:
            None
        """
        self._check_public_access()
        if name is None:
            name = self.name
        if data_format is None:
            data_format = "pytorch_tensor"
        self.path_metadata = "tmp_metadata.json"  # type: ignore
        self.metadata = {
            "name": name,
            "size": size_dataset,
            "input_size": input_size,
            "num_labels": num_labels,
            "task_type": task_type,
            "data_type": data_type,
            "data_format": data_format,
            "comment": comment,
        }
        with open(self.path_metadata, "w") as f:  # type: ignore
            json.dump(self.metadata, f, sort_keys=True, indent=4)

    def _upload(
        self,
        path_data: str,
        path_label: str,
        path_metadata: Union[str, None] = None,
    ) -> None:
        """Uploads a dataset to the cloud.

        Args:
            path_data (str): Path to the data files.
            path_label (str): Path to the label file.
            path_metadata (Optional[str]): Path to the metadata file.

        Raises:
            ValueError: If the dataset already exists in the cloud.

        Returns:
            None
        """
        self._check_public_access()

        # List of existing datasets in the cloud.
        existing_datasets = self.get_existing_datasets()
        if self.name in existing_datasets:
            raise ValueError(
                "Dataset {} already exists in the cloud.".format(self.name)
                + "Available datasets are: {}.".format(existing_datasets)
            )
        if path_metadata is None:
            path_metadata = self.path_metadata
        self._upload_metadata(path_metadata)
        self._upload_data(path_data)
        self._upload_label(path_label)

        # Update dataset list.
        self._update_dataset_list()
