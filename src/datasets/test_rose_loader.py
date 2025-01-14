import unittest
import os
from rose_loader import RoseDatasetLoader


class TestRoseDatasetLoader(unittest.TestCase):
    def setUp(self):
        """
        Set up the loader and load all datasets before running tests.
        """
        self.loader = RoseDatasetLoader()
        self.all_datasets = self.loader.load_all_datasets()
        self.test_file = "test_rose_datasets.json.gz"  # Temporary test file for compressed storage

    def tearDown(self):
        """
        Clean up any temporary files created during testing.
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_dataset_loading(self):
        """
        Test that all datasets are loaded successfully.
        """
        for dataset_name in self.loader.DATASETS_CONFIG:
            self.assertIn(dataset_name["name"], self.all_datasets, f"{dataset_name['name']} not loaded.")

    def test_total_counts(self):
        """
        Verify the total number of sources and ACUs in each dataset.
        """
        for dataset_name, dataset in self.all_datasets.items():
            total_sources = len(dataset)
            total_acus = sum(len(entry["reference_acus"]) for entry in dataset)

            # Check that there are sources and ACUs
            self.assertGreater(total_sources, 0, f"{dataset_name} has no sources.")
            self.assertGreater(total_acus, 0, f"{dataset_name} has no ACUs.")
            print(f"{dataset_name}: {total_sources} sources, {total_acus} ACUs.")

    def test_key_consistency(self):
        """
        Ensure each dataset entry has all required keys.
        """
        required_keys = {"source", "reference", "reference_acus"}

        for dataset_name, dataset in self.all_datasets.items():
            for entry in dataset:
                self.assertTrue(
                    required_keys.issubset(entry.keys()),
                    f"One or more entries in {dataset_name} are missing required keys."
                )

    def test_invalid_dataset_request(self):
        """
        Attempt to fetch a non-existent dataset and confirm a ValueError is raised.
        """
        with self.assertRaises(ValueError):
            self.loader.get_dataset("non_existent_dataset")

    def test_save_and_load_compressed(self):
        """
        Test saving and loading datasets in compressed format.
        """
        # Save the datasets
        self.loader.save_datasets_compressed(self.test_file)
        self.assertTrue(os.path.exists(self.test_file), "Compressed file was not created.")

        # Load the datasets back
        loaded_datasets = self.loader.load_datasets_compressed(self.test_file)

        # Verify that the loaded datasets match the original datasets
        for dataset_name, original_dataset in self.all_datasets.items():
            self.assertIn(dataset_name, loaded_datasets, f"Dataset {dataset_name} missing after loading.")
            self.assertEqual(
                len(original_dataset),
                len(loaded_datasets[dataset_name]),
                f"Dataset {dataset_name} entry count mismatch after loading."
            )

        # Check content equality for a sample entry
        for dataset_name in self.all_datasets.keys():
            original_entry = self.all_datasets[dataset_name][0]
            loaded_entry = loaded_datasets[dataset_name][0]
            self.assertEqual(original_entry, loaded_entry, f"Mismatch in content for dataset {dataset_name}.")

    def test_save_and_load_empty_datasets(self):
        """
        Test saving and loading an empty dataset dictionary.
        """
        empty_loader = RoseDatasetLoader()
        empty_loader.datasets = {}  # Set datasets to empty

        # Save the empty dataset
        empty_loader.save_datasets_compressed(self.test_file)
        self.assertTrue(os.path.exists(self.test_file), "Compressed file for empty datasets was not created.")

        # Load the empty dataset
        loaded_datasets = empty_loader.load_datasets_compressed(self.test_file)
        self.assertEqual(loaded_datasets, {}, "Loaded datasets should be empty.")


if __name__ == "__main__":
    unittest.main()
