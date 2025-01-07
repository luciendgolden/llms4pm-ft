import sys
import os
import torch
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import Dataset
import numpy as np

from llama_fine_tuning_util import load_dataset, print_stats

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
    max_seq_length=128, # Choose any! We auto support RoPE Scaling internally!
    dtype=None,
    load_in_4bit=True,
)

torch.cuda.empty_cache()

FastLanguageModel.for_inference(model)
model.eval()

# T-SAD,A-SAD logic => "ds_labels" is bool
def generate_binary_output(prompt):
    """
    Generates binary predictions (True/False) based on the model's output.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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

def generate_activity_output(prompt, activities) -> str:
    """
    For S-NAP (next activity).
    E.g., returns one of the actual `activities` or "[END]".
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated = tokenizer.batch_decode(outputs.sequences[:, inputs["input_ids"].shape[1]:], 
                                       skip_special_tokens=True)
    text = generated[0]

    # Check if it contains something like "A", "B", "C"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for letter in alphabet:
        if letter in text:
            idx = alphabet.index(letter)
            if idx < len(activities):
                return activities[idx]
            else:
                return "[END]"
    return "[END]"

def compute_y(example):
    """
    Builds a prompt from dataset fields depending on the dataset_task:
      - "TRACE_ANOMALY" → T-SAD uses `trace`
      - "OUT_OF_ORDER"  → A-SAD uses `eventually_follows`
      - "NEXT_ACTIVITY" → S-NAP uses `prefix`
    Then calls generate_binary_output to get "True"/"False".
    """
    if dataset_task == "TRACE_ANOMALY":  # T-SAD
        trace = example["trace"]
        prompt = (
            f"Set of process activities: {example['unique_activities']}\n"
            f"Trace: {trace}\n"
            "Valid: "
        )
        example["pred_label"] = generate_binary_output(prompt)
    elif dataset_task == "OUT_OF_ORDER":  # A-SAD
        ev = example["eventually_follows"]  
        prompt = (
            f"Set of process activities: {example['unique_activities']}\n"
            f"1. Activity: {ev[0]}\n"
            f"2. Activity: {ev[1]}\n"
            "Valid: "
        )
        example["pred_label"] = generate_binary_output(prompt)
    elif dataset_task == "NEXT_ACTIVITY":  # S-NAP
        sorted_acts = sorted(list(example["unique_activities"]))
        prefix = example["prefix"]
        prompt = (
            f"Set of process activities: {sorted_acts}\n"
            f"So far we have executed: {prefix}\n"
            "Next activity: "
        )
        example["pred_label"] = generate_activity_output(prompt, sorted_acts)
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
    Computes micro & macro precision, recall, and F1 scores.
    """
    true_labels = np.array(ds["ds_labels"])
    predicted_labels = np.array(predicted_labels)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='micro', zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )

    metrics = {
        "precision mic": [precision_micro],
        "recall mic": [recall_micro],
        "f1 mic": [f1_micro],
        "precision mac": [precision_macro],
        "recall mac": [recall_macro],
        "f1 mac": [f1_macro],
    }
    
    df = pd.DataFrame(metrics)
    return df


# Fraction Size T-SAD and A-SAD 300k samples, using 10-20% of the dataset should be a good starting point
# Fraction Size S-NAP 60k to 120k samples, using 5-10% of the dataset should be a good starting point

test_set_file = f"test_set_{task_name}.csv"
if not os.path.exists(test_set_file):
    # TODO: adjust frac
    frac = None
    train_ds, val_ds, test_ds = load_dataset(dataset_file, dataset_task, frac=frac)
else:
    # 0.01 frac Total samples: 307
    test_df = pd.read_csv(test_set_file)
    test_ds = Dataset.from_pandas(test_df)

# print the number of samples used for evaluation
print(f"Number of test_ds samples used for evaluation: {len(test_ds)}")
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