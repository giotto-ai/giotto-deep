from gdeep.data import PreprocessingPipeline, Normalisation, \
    PreprocessImageClassification, BasicDataset, \
    DatasetImageClassificationFromFiles, IdentityTransform
from torch.utils.data import Dataset
import os


def test_preeprocessing_pipeline():
    """test preprocessing pipeline"""
    n = Normalisation()
    id = IdentityTransform()
    i = PreprocessImageClassification((32,32))
    file_path = os.path.dirname(os.path.realpath(__file__))

    ds = DatasetImageClassificationFromFiles(
        os.path.join(file_path, "img_data"),
        os.path.join(file_path, "img_data", "labels.csv"))
    # define preprocessing pipeline
    p = PreprocessingPipeline(((i, id, DatasetImageClassificationFromFiles,
                                os.path.join(file_path, "img_data"),
                                os.path.join(file_path, "img_data", "labels.csv")),
                               (Normalisation(), id, BasicDataset)))

    ds2 = DatasetImageClassificationFromFiles(
        os.path.join(file_path, "img_data"),
        os.path.join(file_path, "img_data", "labels.csv"),
        p, dataset=ds)
    ds2[0]
