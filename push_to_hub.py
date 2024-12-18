import argparse
from datasets import Dataset
import json
import os
from tqdm import tqdm

def safe_join(items):
    """
    Safely join items, handling None values and ensuring all items are strings.

    Args:
        items: List or None
    Returns:
        str: Joined string or empty string if input is None/empty
    """
    if not items:  # Handles None or empty list
        return ""
    # Convert all items to strings and filter out None values
    return "\n".join(str(item) for item in items if item is not None)

def upload_to_existing_dataset(all_data, dataset_path="weijiezz/LegalHK"):
    """
    Upload data to an existing Hugging Face dataset.

    Args:
        all_data (list): List of dictionaries containing the data
        dataset_path (str): Path to the existing dataset on Hugging Face Hub
    """
    # Define all fields that should be in the dataset
    processed_data = {
        "plaintiff": [],
        "defendant": [],
        "plaintiff_claim": [],
        "lawsuit_type": [],
        "more_facts": [],
        "related_laws": [],
        "relevant_cases": [],
        "issues": [],
        "court_reasoning": [],
        "judgment_decision": [],
        "support&reject": []
    }

    # Process each data point
    for idx, item in enumerate(tqdm(all_data, desc="Processing data")):
        try:
            # Handle simple string fields with default empty string
            processed_data["plaintiff"].append(str(item.get("plaintiff", "") or ""))
            processed_data["defendant"].append(str(item.get("defendant", "") or ""))
            processed_data["plaintiff_claim"].append(str(item.get("plaintiff_claim", "") or ""))
            processed_data["lawsuit_type"].append(str(item.get("lawsuit_type", "") or ""))
            processed_data["support&reject"].append(str(item.get("support&reject", "") or ""))

            # Handle list fields
            processed_data["more_facts"].append(safe_join(item.get("more_facts", [])))
            processed_data["related_laws"].append(safe_join(item.get("related_laws", [])))
            processed_data["relevant_cases"].append(safe_join(item.get("relevant_cases", [])))
            processed_data["issues"].append(safe_join(item.get("issues", [])))
            processed_data["court_reasoning"].append(safe_join(item.get("court_reasoning", [])))
            processed_data["judgment_decision"].append(safe_join(item.get("judgment_decision", [])))

        except Exception as e:
            print(f"Error processing record {idx}:")
            print(f"Error message: {str(e)}")
            print(f"Problematic record: {item}")
            continue

    # Create Dataset object
    dataset = Dataset.from_dict(processed_data)

    # Push to hub
    dataset.push_to_hub(dataset_path)

    print(f"\nSuccessfully uploaded data to {dataset_path}")
    print(f"Dataset statistics:")
    print(f"Number of records: {len(all_data)}")
    print("\nSample record structure:")
    for key in processed_data.keys():
        print(f"{key}: {type(processed_data[key][0]).__name__}")

def main(args):
    """
    Main function to handle data loading and upload.
    """
    base_path = args.base_path
    all_data = []

    # Collect data using the existing file walking logic
    for root, dirs, _ in os.walk(base_path):
        if 'EN' in dirs:
            input_dir = os.path.join(root, 'EN')
            for file_name in tqdm(os.listdir(input_dir), desc="Loading files"):
                if file_name.endswith('.json'):
                    input_path = os.path.join(input_dir, file_name)
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            all_data.append(content)
                    except Exception as e:
                        print(f"Error loading file {file_name}: {str(e)}")
                        continue

    print(f"Loaded {len(all_data)} records")

    # Validate data structure
    print("\nValidating data structure...")
    required_fields = [
        "plaintiff", "defendant", "plaintiff_claim", "lawsuit_type",
        "more_facts", "related_laws", "relevant_cases", "issues",
        "court_reasoning", "judgment_decision", "support&reject"
    ]

    for idx, item in enumerate(all_data):
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            print(f"Warning: Record {idx} is missing fields: {missing_fields}")

    # Upload to the existing dataset
    if all_data:
        upload_to_existing_dataset(all_data)
    else:
        print("No data to upload!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="filter_processed_data")


    args = parser.parse_args()
    main(args)