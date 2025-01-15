import pytest
from datasets import load_dataset as original_load_dataset


def trusted_load_dataset(*args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    return original_load_dataset(*args, **kwargs)


@pytest.fixture(autouse=True)
def patch_load_dataset(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", trusted_load_dataset)
