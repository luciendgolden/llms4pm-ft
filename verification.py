# verification.py

import sys
import os
from functools import partial

import torch
from transformers import AutoTokenizer

from llama_fine_tuning_util import load_dataset
from prompt_tokenize import instruction_map_fn

def verify_samples(task_name, num_samples=3):
    """
    Verifies a subset of the dataset by decoding inputs and labels to ensure correctness.

    Args:
        task_name (str): The name of the task to verify (e.g., "T-SAD", "A-SAD", "S-NAP").
        num_samples (int): Number of samples to verify.
    """
    task_to_dataset = {
        "T-SAD": ("data/T_SAD.csv", "TRACE_ANOMALY"),
        "A-SAD": ("data/A_SAD.csv", "OUT_OF_ORDER"),
        "S-NAP": ("data/S_NAP.csv", "NEXT_ACTIVITY"),
    }
    
    if task_name not in task_to_dataset:
        raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {list(task_to_dataset.keys())}")
    
    dataset_file, task_type = task_to_dataset[task_name]
    
    print(f"\nLoading dataset for task '{task_name}' with task type '{task_type}'...")
    
    # use frac=0.1 or None if you want the full dataset
    frac = 0.1
    raw_train_dataset, raw_val_dataset, raw_test_dataset = load_dataset(dataset_file, task_type, frac)
    train_dataset_size = len(raw_train_dataset)
    print(f"Loaded {train_dataset_size} training samples, {len(raw_val_dataset)} validation samples, and {len(raw_test_dataset)} test samples.")
    
    # Initialize tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token="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",
    )
    
    map_func = partial(instruction_map_fn, tokenizer=tokenizer, task_type=task_type)
    columns_to_remove = ["model_id", "revision_id", "id", "num_unique_activities"]
    
    print("\nApplying mapping function to the training dataset...")
    train_ds = raw_train_dataset.map(map_func, batched=True, remove_columns=columns_to_remove)
    print("Mapping applied to training dataset.")
    
    # select the first n samples to verify
    samples_to_verify = train_ds.select(range(min(num_samples, len(train_ds))))
    
    print(f"\nVerifying the first {len(samples_to_verify)} samples from the training dataset...")
    
    for i, sample in enumerate(samples_to_verify):
        print(f"\n--- Sample {i+1} ---")
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # remove -100 tokens from the label tokens
        decoded_label_tokens = [token for token in labels if token != -100]
        decoded_label = tokenizer.decode(decoded_label_tokens, skip_special_tokens=True)
        
        print("Decoded Input:")
        print(decoded_input)
        print("Decoded Label:")
        print(decoded_label)
        
        if task_type in ["TRACE_ANOMALY", "OUT_OF_ORDER"]:
            ds_label = sample.get('ds_labels', None)
            if ds_label is None:
                expected_label = "Unknown"
                print(">>> Warning: 'ds_labels' field is missing in the sample.")
            else:
                if task_type in ["TRACE_ANOMALY", "OUT_OF_ORDER"]:
                    expected_label = "True" if ds_label else "False"
                elif task_type == "NEXT_ACTIVITY":
                    pass
                else:
                    expected_label = "Unknown"

            print("Expected Label:")
            print(expected_label)
            print("Label Match:", expected_label == decoded_label)

            if expected_label != decoded_label:
                print(">>> Warning: Label mismatch detected!")
            else:
                print("Label correctly assigned.")
        elif task_type == "NEXT_ACTIVITY":
            prefix = raw_train_dataset[i]["prefix"]
            next_act = raw_train_dataset[i]["next"]
            
            print("\nExpected Next Activity:")
            print(next_act)
            match = (next_act.strip().lower() == decoded_label.strip().lower())
            print("Label Match:", match)
            if not match:
                print(">>> Warning: Next-activity mismatch detected!")
            else:
                print("Next-activity label is correctly assigned.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verification.py <TASK_NAME> [NUM_SAMPLES]")
        print("Example: python verification.py A-SAD 3")
        sys.exit(1)
    
    task_name = sys.argv[1]
    
    if len(sys.argv) >= 3:
        try:
            num_samples = int(sys.argv[2])
        except ValueError:
            print("NUM_SAMPLES must be an integer.")
            sys.exit(1)
    else:
        num_samples = 3
    
    verify_samples(task_name, num_samples)
