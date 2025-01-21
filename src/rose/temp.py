import json
import gzip

def rename_key(data, old_key, new_key):
    """
    Recursively rename keys in a JSON-like structure.
    """
    if isinstance(data, dict):
        # Rename the key if it matches old_key
        return {
            (new_key if k == old_key else k): rename_key(v, old_key, new_key)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        # Recurse into lists
        return [rename_key(item, old_key, new_key) for item in data]
    else:
        # Return the data as is if it's neither dict nor list
        return data


if __name__ == "__main__":
    # File paths
    input_file = "rose_datasets.json.gz"  # Replace with your input file path
    output_file = "rose_datasets.json.gz"  # Replace with your desired output file path

    # Read the compressed JSON file
    with gzip.open(input_file, "rt", encoding="utf-8") as file:
        json_data = json.load(file)

    # Rename the key
    updated_data = rename_key(json_data, "system_claims_gpt35", "system_claims_gpt-3.5-turbo")

    # Save the updated JSON back to a compressed file
    with gzip.open(output_file, "wt", encoding="utf-8") as file:
        json.dump(updated_data, file, indent=2)

    print(f"Key renamed and output saved to {output_file}.")
