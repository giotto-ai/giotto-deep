class DatasetImageClassificationFromFiles(TransformableDataset):
    """This class is useful to build a dataset
    directly from image files
    
    Args:
        img_folder (string):
            The path to the folder where the training
            images are located
        labels_file (string):
            The path and file name of the labels.
            It shall be a ``.csv`` file with two columns.
            The first columns contains the name of the
            image and the second one contains the
            label value
        transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``
        target_transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``
    """
    def __init__(self, img_folder: str=".",
                 labels_file:str="labels.csv",
                 transform=None,
                 target_transform=None,
                 dataset=None  # only for pipeline compatibility
                 ) -> None:

        super().__init__(transform=transform,
                         target_transform=target_transform)
        self.img_folder = img_folder
        self.img_labels = pd.read_csv(labels_file)
        if dataset:
            self.transform.fit_to_data(dataset)
            self.target_transform.fit_to_data(dataset)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self._get_image(idx)

        label = self.img_labels.iloc[idx, 1]
        imageT = self.transform(image)
        label = self.target_transform(label)
        image.close()
        return imageT, label

    def _get_image(self, idx:int):
        img_path = os.path.join(self.img_folder, self.img_labels.iloc[idx, 0])
        try:
            image = Image.open(img_path)
        except UnidentifiedImageError:
            warnings.warn(f"The image {img_path} canot be loaded. Skipping it.")
            return None, None
        return image


class DatasetFromArray(TransformableDataset):
    """This class is useful to build dataloaders
    from a array of X and y. Tensors are also supported.

    Args:
        X (np.array or torch.Tensor):
            The data. The first dimension is the datum index
        y (np.array or torch.Tensor):
            The labels, need to match the first dimension
            with the data

    """
    def __init__(self, X: Union[Tensor, np.ndarray],
                 y: Union[Tensor, np.ndarray],
                 transform=None, target_transform=None,
                 dataset=None # for compatiblitiy reasons
                 ) -> None:
        super().__init__(transform=transform,
                         target_transform=target_transform)
        self.X = self._from_numpy(X)
        y = self._from_numpy(y)
        self.y = self._long_or_float(y)
        self.transform.fit_to_data(X)
        self.target_transform.fit_to_data(y)

    @staticmethod
    def _from_numpy(X):
        """this is torch.from_numpy() that also allows
        for tensors"""
        if isinstance(X, torch.Tensor):
            return X
        return torch.from_numpy(X)

    @staticmethod
    def _long_or_float(y):
        if isinstance(y, torch.Tensor):
            return y
        if isinstance(y, np.float16) or isinstance(y, np.float32) or isinstance(y, np.float64):
            return torch.tensor(y).float()
        return torch.tensor(y).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        y = self.target_transform(self.y[idx])
        X = self.transform(self.X[idx])
        return X, y


class DlBuilderFromDataCloud(AbstractDataLoaderBuilder):
    """Class that loads data from Google Cloud Storage
    
    This class is useful to build dataloaders from a dataset stored in
    the GDeep Dataset Cloud on Google Cloud Storage.

    The constructor takes the name of a dataset as a string, and a string
    for the download directory. The constructor will download the dataset
    to the download directory. The dataset is downloaded in the version
    used by Datasets Cloud, which may be different from the version
    used by the dataset's original developers.
    
    Args:
        dataset_name (str):
            The name of the dataset.
        download_dir (str):
            The directory where the dataset will be downloaded.
        use_public_access (bool):
            Whether to use public access. If you want
            to use the Google Cloud Storage API, you must set this to True.
            Please make sure you have the appropriate credentials.
        path_to_credentials (str):
            Path to the credentials file.
            Only used if public_access is False and credentials are not
            provided. Defaults to None.
        

    Returns:
        torch.utils.data.DataLoader: The dataloader for the dataset.

    Raises:
        ValueError:
            If the dataset_name is not a valid dataset that exists
            in Datasets Cloud.
        ValueError:
            If the download_directory is not a valid directory.
    """
    def __init__(self,
                 dataset_name: str,
                 download_directory: str,
                 use_public_access: bool=True,
                 path_to_credentials: Union[None, str] = None,
                 ):
        self.dataset_name = dataset_name
        self.download_directory = download_directory
        
        # Download the dataset if it does not exist
        self.download_directory
        
        self._download_dataset(use_public_access=use_public_access, 
                               path_to_credentials = path_to_credentials)

        self.dl_builder = None
        
        # Load the metadata of the dataset
        with open(join(self.download_directory, self.dataset_name,
                       "metadata.json")) as f:
            self.dataset_metadata = json.load(f)
            
        # Load the data and labels of the dataset
        if self.dataset_metadata['data_type'] == 'tabular':
            if self.dataset_metadata['data_format'] == 'pytorch_tensor':
                data = torch.load(join(self.download_directory, 
                                       self.dataset_name, "data.pt"))
                labels = torch.load(join(self.download_directory, 
                                         self.dataset_name, "labels.pt"))

                self.dl_builder = BuildDataLoaders((DatasetFromArray(data, labels),))
            elif self.dataset_metadata['data_format'] == 'numpy_array':
                data = np.load(join(self.download_directory,
                                    self.dataset_name, "data.npy"))
                labels = np.load(join(self.download_directory,
                                      self.dataset_name, "labels.npy"))
                self.dl_builder = BuildDataLoaders((DatasetFromArray(data, labels),))
            else:
                raise ValueError("Data format {}"\
                    .format(self.dataset_metadata['data_format']) +
                                 "is not yet supported.")
        else:
            raise ValueError("Dataset type {} is not yet supported."\
                .format(self.dataset_metadata['data_type']))

    def _download_dataset(self,
                          path_to_credentials: Union[None, str] =None,
                          use_public_access: bool=True,) -> None:
        """Only download if the download directory does not exist already
        and if download directory contains at least three files (metadata,
        data, labels).
        
        Args:
            path_to_credentials (str):
                Path to the credentials file.
            use_public_access (bool):
                Whether to use public access. If you want
                to use the Google Cloud Storage API, you must set this to True.
                
        Returns:
            None
        """
        if (not os.path.isdir(join(self.download_directory, self.dataset_name))
                            or len(os.listdir(join(self.download_directory,
                                            self.dataset_name))) < 3):
            # Delete the download directory if it exists but does not contain
            # the wanted number of files
            if (os.path.isdir(join(self.download_directory, self.dataset_name))
                and
                len(os.listdir(join(self.download_directory,
                                   self.dataset_name))) < 3): # type: ignore
                print("Deleting the download directory because it does "+
                      "not contain the dataset")
                shutil.rmtree(self.download_directory, ignore_errors=True)
                
            print("Downloading dataset {} to {}"\
                    .format(self.dataset_name, self.download_directory))
            dataset_cloud = DatasetCloud(self.dataset_name,
                                    download_directory=self.download_directory,
                                    path_to_credentials=path_to_credentials,
                                    use_public_access = use_public_access,
                                    )
            dataset_cloud.download()
            del dataset_cloud
        else:
            print("Dataset '%s' already downloaded" % self.dataset_name)

    def get_metadata(self) -> Dict[str, Any]:
        """ Returns the metadata of the dataset.
        
        Returns:
            Dict[str, Any]:
                The metadata of the dataset.
        """
        return self.dataset_metadata
    
    def build_dataloaders(self, **kwargs)\
        -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Builds the dataloaders for the dataset.
        
        Args:
            **kwargs: Arguments for the dataloader builder.
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]:
                The dataloaders for the dataset (train, validation, test).
        """
        return self.dl_builder.build_dataloaders(**kwargs) # type: ignore