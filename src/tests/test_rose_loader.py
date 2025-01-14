import os
import pytest

from src.rose.rose_loader import RoseDatasetLoader


@pytest.fixture
def rose_loader():
    """
    Fixture to initialize the RoseDatasetLoader and load all datasets.
    """
    loader = RoseDatasetLoader()
    all_datasets = loader.load_all_datasets()
    return loader, all_datasets


@pytest.fixture
def test_file():
    """
    Fixture for a temporary test file. Ensures cleanup after tests.
    """
    filename = "test_rose_datasets.json.gz"
    yield filename
    if os.path.exists(filename):
        os.remove(filename)


def test_dataset_loading(rose_loader):
    """
    Test that all datasets are loaded successfully.
    """
    loader, all_datasets = rose_loader
    for dataset_name in loader.DATASETS_CONFIG:
        assert dataset_name["name"] in all_datasets, f"{dataset_name['name']} not loaded."


def test_total_counts(rose_loader):
    """
    Verify the total number of sources and ACUs in each dataset.
    """
    _, all_datasets = rose_loader
    for dataset_name, dataset in all_datasets.items():
        total_sources = len(dataset)
        total_acus = sum(len(entry["reference_acus"]) for entry in dataset)

        # Check that there are sources and ACUs
        assert total_sources > 0, f"{dataset_name} has no sources."
        assert total_acus > 0, f"{dataset_name} has no ACUs."
        print(f"{dataset_name}: {total_sources} sources, {total_acus} ACUs.")


def test_key_consistency(rose_loader):
    """
    Ensure each dataset entry has all required keys.
    """
    _, all_datasets = rose_loader
    required_keys = {"source", "reference", "reference_acus"}

    for dataset_name, dataset in all_datasets.items():
        for entry in dataset:
            assert required_keys.issubset(entry.keys()), \
                f"One or more entries in {dataset_name} are missing required keys."


def test_invalid_dataset_request(rose_loader):
    """
    Attempt to fetch a non-existent dataset and confirm a ValueError is raised.
    """
    loader, _ = rose_loader
    with pytest.raises(ValueError):
        loader.get_dataset("non_existent_dataset")


def test_save_and_load_compressed(rose_loader, test_file):
    """
    Test saving and loading datasets in compressed format.
    """
    loader, all_datasets = rose_loader

    # Save the datasets
    loader.save_datasets_compressed(test_file)
    assert os.path.exists(test_file), "Compressed file was not created."

    # Load the datasets back
    loaded_datasets = loader.load_datasets_compressed(test_file)

    # Verify that the loaded datasets match the original datasets
    for dataset_name, original_dataset in all_datasets.items():
        assert dataset_name in loaded_datasets, f"Dataset {dataset_name} missing after loading."
        assert len(original_dataset) == len(loaded_datasets[dataset_name]), \
            f"Dataset {dataset_name} entry count mismatch after loading."

    # Check content equality for a sample entry
    for dataset_name in all_datasets.keys():
        original_entry = all_datasets[dataset_name][0]
        loaded_entry = loaded_datasets[dataset_name][0]
        assert original_entry == loaded_entry, f"Mismatch in content for dataset {dataset_name}."


def test_save_and_load_empty_datasets(test_file):
    """
    Test saving and loading an empty dataset dictionary.
    """
    loader = RoseDatasetLoader()
    loader.datasets = {}  # Set datasets to empty

    # Save the empty dataset
    loader.save_datasets_compressed(test_file)
    assert os.path.exists(test_file), "Compressed file for empty datasets was not created."

    # Load the empty dataset
    loaded_datasets = loader.load_datasets_compressed(test_file)
    assert loaded_datasets == {}, "Loaded datasets should be empty."
