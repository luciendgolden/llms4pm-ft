import ast
import re
import sys
import os
import torch
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import Dataset
import numpy as np

from llama_fine_tuning_util import load_dataset, print_stats, setify

tqdm.pandas()

task_to_dataset = {
    "T-SAD": ("data/T_SAD.csv", "TRACE_ANOMALY"),
    "A-SAD": ("data/A_SAD.csv", "OUT_OF_ORDER"),
    "S-NAP": ("data/S_NAP.csv", "NEXT_ACTIVITY"),
}

model_dir = sys.argv[1]
task_name = sys.argv[2]

if task_name not in task_to_dataset:
    raise ValueError(f"Unsupported task: {task_name}. "
                        f"Choose from {list(task_to_dataset.keys())}")
    
dataset_file, dataset_task = task_to_dataset[task_name]

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=5, # Choose any! We auto support RoPE Scaling internally!
    dtype=None,
    load_in_4bit=True,
)

torch.cuda.empty_cache()
FastLanguageModel.for_inference(model)
model.eval()

def parse_malformed_list(value):
    try:
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            items = re.findall(r"'([^']*)'", value)
            return items
        return value
    except Exception as e:
        print(f"Error parsing value: {value} -> {e}")
        return None

# T-SAD,A-SAD logic => "ds_labels" is bool
def generate_binary_output(prompt):
    """
    Generates binary predictions (True/False) based on the model's output.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
        )
    decoded = tokenizer.batch_decode(output_tokens.sequences, skip_special_tokens=True)
    decoded_text = decoded[0].strip().lower()
    if "true" in decoded_text:
        return True
    elif "false" in decoded_text:
        return False
    
    #print(decoded[0], "neither Yes nor No")
    #print(tokenizer.decode(output_tokens[0][0]))
    return False

def generate_next_activity(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
        )
    decoded = tokenizer.batch_decode(output_tokens.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded[0].strip()

def compute_y(example):
    """
    Builds a prompt from dataset fields depending on the dataset_task:
      - "TRACE_ANOMALY" → T-SAD uses `trace`
      - "OUT_OF_ORDER"  → A-SAD uses `eventually_follows`
      - "NEXT_ACTIVITY" → S-NAP uses `prefix`
    """
    if dataset_task == "TRACE_ANOMALY":  # T-SAD
        trace = example["trace"]
        prompt = (
            f"Set of process activities: {example['unique_activities']}\n"
            f"Trace: {trace}\n"
            "Valid: "
        )
        example["pred_label"] = generate_binary_output(prompt)
        #print(f"prompt: {prompt}, pred_label: {example['pred_label']}")
    elif dataset_task == "OUT_OF_ORDER":  # A-SAD
        ev = example["eventually_follows"]  
        prompt = (
            f"Set of process activities: {example['unique_activities']}\n"
            f"1. Activity: {ev[0]}\n"
            f"2. Activity: {ev[1]}\n"
            "Valid: "
        )
        example["pred_label"] = generate_binary_output(prompt)
        #print(f"prompt: {prompt}, pred_label: {example['pred_label']}")
    elif dataset_task == "NEXT_ACTIVITY":  # S-NAP
        local_acts = sorted(set(list(example["unique_activities"])))
        if "[END]" not in local_acts:
            local_acts.append("[END]")

        prefix = example["prefix"]
        prompt = (
            f"List of activities: {local_acts}\n"
            f"Sequence of activities: {prefix}\n"
            "Next activity: "
        )
        example["pred_label"] = generate_next_activity(prompt)
    else:
        example["pred_label"] = None

    return example


def run_predictions(ds: Dataset) -> list:
    """
    Applies compute_y to each example in ds, returning the predicted labels.
    """
    ds = ds.map(compute_y, desc="Generating predictions")
    return ds["pred_label"]

def evaluate(ds: Dataset, predicted_labels: list):
    """
    Compute metrics for T-SAD / A-SAD (boolean classification) or for S-NAP (string match).
    """
    if dataset_task in ["TRACE_ANOMALY", "OUT_OF_ORDER"]:
        true_labels = np.array(ds["ds_labels"])
        predicted_labels = np.array(predicted_labels)

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='micro', zero_division=0
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro', zero_division=0
        )

        return pd.DataFrame({
            "precision mic": [precision_micro],
            "recall mic": [recall_micro],
            "f1 mic": [f1_micro],
            "precision mac": [precision_macro],
            "recall mac": [recall_macro],
            "f1 mac": [f1_macro],
        })
    elif dataset_task == "NEXT_ACTIVITY":
        gold_ids, pred_ids = [], []
        for i in range(len(ds)):
            local_acts = sorted(set(list(ds[i]["unique_activities"])))
            if "[END]" not in local_acts:
                local_acts.append("[END]")

            gold = ds[i]["next"]
            pred = predicted_labels[i]
            
            try:
                gold_idx = local_acts.index(gold)
            except ValueError:
                gold_idx = -1
            try:
                pred_idx = local_acts.index(pred)
            except ValueError:
                pred_idx = -1
                
            print(f"local_acts: {local_acts}\n")
            print(f"gold: {gold}, pred: {pred}\n")
            print(f"gold_idx: {gold_idx}, pred_idx: {pred_idx}\n")
            print("--------------------\n")

            gold_ids.append(gold_idx)
            pred_ids.append(pred_idx)

        gold_arr = np.array(gold_ids)
        pred_arr = np.array(pred_ids)
        mask = (gold_arr != -1) & (pred_arr != -1)
        gold_arr = gold_arr[mask]
        pred_arr = pred_arr[mask]

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            gold_arr, pred_arr, average='micro', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gold_arr, pred_arr, average='macro', zero_division=0
        )
        return pd.DataFrame({
            "precision mic": [precision_micro],
            "recall mic": [recall_micro],
            "f1 mic": [f1_micro],
            "precision mac": [precision_macro],
            "recall mac": [recall_macro],
            "f1 mac": [f1_macro],
        })
        
# ------------- Main ----------------

# TODO: adjust frac if you want to test on a smaller fraction of the dataset
frac = 0.1
_, _, test_ds = load_dataset(dataset_file, dataset_task, frac=frac)

# print stats used for test
print_stats("Test", test_ds)

# Predict on test dataset
predicted_labels = run_predictions(test_ds)

# Compute metrics
results_df = evaluate(test_ds, predicted_labels)
print("Metrics:\n", results_df)

# Save results to a CSV file
model_subdir = f"eval"
os.makedirs(model_subdir, exist_ok=True)
csv_name = os.path.join(model_subdir, f"finetune_eval_{task_name}_{model_dir.replace('/', '_')}_results.csv")
results_df.to_csv(csv_name, index=False)
print(f"Saved results to {csv_name}")