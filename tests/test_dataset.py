import pytest

from pysurv.data import Dataset


def test_dataset():
    dataset = Dataset("measurements", "controls", "stations")
    assert isinstance(dataset, Dataset)
