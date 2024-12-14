import json
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from jinja2 import Template, Environment, FileSystemLoader
import torch
from pprint import pprint

if __name__ == "__main__":
    input_output_pairs = []
    base_path = "refine_processed_data"
    for root, dirs, _ in os.walk(base_path):
        if 'EN' in dirs:
            input_dir = os.path.join(root, 'EN')
            relative_path = os.path.relpath(input_dir, base_path)
            output_dir = os.path.join('filter_processed_data', relative_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            input_output_pairs.append((input_dir, output_dir))

    all_data = []
    all_count = 0
    for input_dir, output_dir in tqdm(input_output_pairs, desc="Processing files", total=len(input_output_pairs)):
        for file_name in tqdm(os.listdir(input_dir), desc="Processing list", total=len(os.listdir(input_dir))):
            if file_name.endswith('.json'):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(file_name)[0]}.json"
                )
                with (open(input_path, 'r', encoding='utf-8') as f):
                    all_count += 1
                    try:
                        content = json.load(f)
                        # pprint(content)
                        # exit()
                        if content.get("plaintiff_claim")!="" and content["plaintiff_claim"] and len(content.get("issues"))>0 and len(content.get("more_facts"))>0 and \
                            len(content.get("court_reasoning"))>0 and \
                            len(content.get("judgment_decision"))>0 and \
                                (content.get("support&reject").lower()=="support" or content.get("support&reject").lower()=="reject"):
                            all_data.append([input_dir,output_path, content])
                        else:
                            print(f"Skipped {input_path}")
                            continue
                    except Exception as e:
                        print(f"Error loading {input_path}: {e}")
                        continue

    # save results
    for input_dir,output_path, content in all_data:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

    # Logging statistics
    print(f"number of all_count: {all_count}")
    print(f"number of results: {len(all_data)}")
    print(f"number of invalid results: {len(input_output_pairs)-len(all_data)}")