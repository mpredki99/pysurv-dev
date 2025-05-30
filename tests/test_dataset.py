import pytest

from pysurv.data import Dataset

def test_dataset():
    dataset = Dataset()
    assert isinstance(dataset, Dataset)
