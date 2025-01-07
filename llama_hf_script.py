"""
This script fine-tunes a pre-trained language model especially autoregressive language model (CLM)
such as GPT-2, GPT-3, or Meta-Llama on a custom task. The task can be one of the following:

- T-SAD: Given a trace σ, decide if σ is a valid execution of the underlying process or not, without knowing the behavior allowed in the process. Each row contains a trace (column trace) with a corresponding label (column anomalous) indicating whether the trace represents a valid execution of the underlying process. The set of activities that can occur in the process are also given (column unique_activities).

model_id,revision_id,trace,label,unique_activities,anomalous,id
c78bef3bc4f043e880c51a5de86f7b33,cf03653c9c664b55a18da5b53ca9cee5,"['Take comprehensive exam', 'Submit course form (at least ECTS)', 'Complete courses', 'Get an international publication', 'Follow seminar on research methodology', 'Give first doctoral seminar', 'Participate in international conference', 'Give second doctoral seminar']",False,"{'Take comprehensive exam', 'Submit course form (at least ECTS)', 'Complete courses', 'Follow seminar on research methodology', 'Give first doctoral seminar', 'Get an international publication', 'Give second doctoral seminar', 'Participate in international conference'}",False,c78bef3bc4f043e880c51a5de86f7b33_cf03653c9c664b55a18da5b53ca9cee5

- A-SAD: Given an eventually-follows relation ef = a ≺ b of a trace σ, decide if ef represents a valid execution order of the two activities a and b that are executed in a process or not, without knowing the behavior allowed in the process.
Each row contains an eventually-follows relation (column eventually_follows) with a corresponding label (column out_of_order) indicating wether the two activities of the relation were executed in an invalid order (TRUE) or in a valid order (FALSE) according to the underlying process (model). The set of activities that can occur in the process are also given (column unique_activities).

model_id,revision_id,out_of_order,unique_activities,eventually_follows,id
2b4e4aca49ef4694a290b956fe18eb9b,f9f65a9604b4434996eede7b550b8f8a,True,"{'Register claim', 'Perform assessment', 'Phone garage to authorise repairs', 'Send letter', 'Checks insurance claim', 'Reject claim', 'Schedule payment', 'Check document'}","('Phone garage to authorise repairs', 'Reject claim')",2b4e4aca49ef4694a290b956fe18eb9b_f9f65a9604b4434996eede7b550b8f8a

- S-NAP: Given an event log L and a prefix p_k of length k, with 1 < k, predict the next activity a_k+1
Each row contains a trace prefix (column prefix) with a corresponding next activity (column next) indicating the activity that should be performed next after the last activity of the prefix  according to the trace from which the prefix was generated. The set of activities that can occur in the process are also given (column unique_activities).

model_id,revision_id,trace,prefix,next,unique_activities,id
f59a5a5a07b64916bcbd843e48485c0e,11c2f63f1f684c9dabbdb18d5e47bcca,"['mold upper and lower part of the enginge', 'bend front defender', 'wield parts together', 'bend bars for the frame', 'insert outlets and cylinders', 'make seat', 'bend rear defender', 'weld bars together', 'assemble parts']","['mold upper and lower part of the enginge', 'bend front defender', 'wield parts together']",bend bars for the frame,"{'bend bars for the frame', 'weld bars together', 'insert outlets and cylinders', 'bend rear defender', 'wield parts together', 'bend front defender', 'mold upper and lower part of the enginge', 'assemble parts', 'make seat'}",f59a5a5a07b64916bcbd843e48485c0e_11c2f63f1f684c9dabbdb18d5e47bcca

On a 40GB A100
- Consider 8-bit or 16-bit instead of 4-bit quantization if you want more speed and (possibly) better accuracy.
- Disable gradient checkpointing to avoid extra forward passes.
- Turn off short_run for a real run.
- (Optionally) reduce logging overhead and/or do fewer evaluation steps.
- Possibly try xformers/flash attention for even faster performance.
"""
import os
import sys
import time
import copy
import datetime
import pandas as pd
from functools import partial

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    IntervalStrategy,
    Trainer,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from sklearn.metrics import precision_recall_fscore_support
from llama_fine_tuning_util import load_dataset, print_stats, format_time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.backends.cudnn.benchmark = True

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.is_available())
print(torch.cuda.memory_summary())

if not torch.cuda.is_available():
    print("Warning: No GPU detected. Training might be very slow.")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
#-------------------------TASK SELECTION-----------------------------------------------------------

task_name = sys.argv[1]

task_to_dataset = {
    "T-SAD": ("data/T_SAD.csv", "TRACE_ANOMALY"),
    "A-SAD": ("data/A_SAD.csv", "OUT_OF_ORDER"),
    "S-NAP": ("data/S_NAP.csv", "NEXT_ACTIVITY"),
}

if task_name not in task_to_dataset:
    raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {list(task_to_dataset.keys())}")

dataset_file, task_type = task_to_dataset[task_name]

#-------------------------LOAD THE DATA----------------------------------------------
"""
Load the training and validation datasets using the load_trace_data function.
"""
# fraction of the dataset to use for training and validation
# TODO: Set to None for "all" production
frac = 0.1

raw_train_dataset, raw_val_dataset, raw_test_dataset = load_dataset(dataset_file, task_type, frac)
train_dataset_size = len(raw_train_dataset)

test_output_file = f"test_set_{task_name}.csv"
raw_test_dataset.to_csv(test_output_file, index=False)
print(f"Test set saved to {test_output_file}")

#--------------------------INITIALIZE VARIABLES-----------------------------------------------------------
"""
Set up all the key variables for training, such as model parameters, dataset sample sizes, 
number of epochs, learning rate, batch sizes, and directory for saving checkpoints and logs.
"""
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
max_seq_length = 5 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
train_batch_size = 2
gradient_accumulation_steps = 16
# TODO: Set to 3 for llama production
epochs = 3

optimizer = "adamw_8bit"
label_model_name = model_name.replace("/", "_")

train_samples = "all"
valid_samples = "all"

learning_rate = 1e-5
eval_batch_size = 4
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_steps = 50

# min max bounds
MIN_EVAL_STEPS = 50 
MAX_EVAL_STEPS = 1000
MIN_SAVE_STEPS = 50
MAX_SAVE_STEPS = 1000

# dynamic eval_steps
num_updates_per_epoch = train_dataset_size // (train_batch_size * gradient_accumulation_steps)
if train_dataset_size < 10_000:
    desired_evals_per_epoch = 2
elif 10_000 <= train_dataset_size <= 100_000:
    desired_evals_per_epoch = 6
else:
    desired_evals_per_epoch = 12
    
eval_steps = max(1, num_updates_per_epoch // desired_evals_per_epoch)
eval_steps = max(MIN_EVAL_STEPS, min(eval_steps, MAX_EVAL_STEPS))

# dynamic save_steps
if train_dataset_size < 10_000:
    save_steps = eval_steps * 2 
elif 10_000 <= train_dataset_size <= 100_000:
    save_steps = eval_steps
else:
    save_steps = max(MIN_SAVE_STEPS, eval_steps // 2)
    save_steps = min(save_steps, MAX_SAVE_STEPS)

save_total_limit = 5

# TODO: Short run for throughput estimation
short_run = False

save_dir = (f"{label_model_name}_{task_name}_samples-{train_samples}_epochs-{epochs}_"
            f"lr-{learning_rate}_batch-{train_batch_size}x{gradient_accumulation_steps}_time-{timestamp}")


#-------------------------INITIALIZE MODEL AND TOKENIZER-------------------------
"""
Load the pre-trained model and tokenizer using the FastLanguageModel wrapper. 
Enable options like sequence length, precision, and quantization.
"""
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="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",
)
model = FastLanguageModel.get_peft_model(
    model,
    # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    r=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    # Supports any, but = 0 is optimized
    lora_dropout=0,
    # Supports any, but = "none" is optimized
    bias="none",
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=False,
    random_state=3407,
    # We support rank stabilized LoRA
    use_rslora=False,
    # And LoftQ
    loftq_config=None,
)

#-------------------------BUILD THE PROMPT AND TOKENIZE INPUT-------------------------
"""
Define functions to prepare the training data by creating prompts and tokenizing inputs/labels.
"""
def build_instruction_encoding(example, tokenizer, max_length=512, task_type="TRACE_ANOMALY"):
    """
    Returns a dict with input_ids, attention_mask, and labels
    so that the prompt is masked out (label = -100)
    and only the final 'answer' tokens are labeled.
    """
    ds_label = example["ds_labels"]
    if task_type == "TRACE_ANOMALY":
        label_str = "True" if ds_label else "False"
        prompt_text = f"Set of process activities: {set(example['unique_activities'])}\nTrace: {example['trace']}\nValid: "
    elif task_type == "OUT_OF_ORDER":
        label_str = "True" if ds_label else "False"
        eventually = example["eventually_follows"]
        prompt_text = f"Set of process activities: {set(example['unique_activities'])}\n1. Activity: {eventually[0]}\n2. Activity: {eventually[1]}\nValid: "
    elif task_type == "NEXT_ACTIVITY":
        sorted_acts = sorted(list(example["unique_activities"]))
        label_idx = ds_label - 1
        label_str = "[END]" if (label_idx < 0 or label_idx >= len(sorted_acts)) else sorted_acts[label_idx]
        prompt_text = f"Set of process activities: {sorted_acts}\nSo far we have executed: {example['prefix']}\nNext activity: "
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    enc_prompt = tokenizer(prompt_text, add_special_tokens=False)
    enc_answer = tokenizer(label_str, add_special_tokens=False)
    input_ids = enc_prompt["input_ids"] + enc_answer["input_ids"]
    attention_mask = enc_prompt["attention_mask"] + enc_answer["attention_mask"]
    labels = ([-100] * len(enc_prompt["input_ids"])) + enc_answer["input_ids"]
    
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]
    
    return {"input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels}

def instruction_map_fn(examples, tokenizer, task_type):
    """
    Map function to tokenize inputs and labels for the training dataset.
    """
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["ds_labels"])):
        record = {"ds_labels": examples["ds_labels"][i], "unique_activities": examples["unique_activities"][i]}
        if task_type == "TRACE_ANOMALY":
            record["trace"] = examples["trace"][i]
        elif task_type == "OUT_OF_ORDER":
            record["eventually_follows"] = examples["eventually_follows"][i]
        elif task_type == "NEXT_ACTIVITY":
            record["prefix"] = examples["prefix"][i]
        out = build_instruction_encoding(record, tokenizer=tokenizer, max_length=512, task_type=task_type)
        results["input_ids"].append(out["input_ids"])
        results["attention_mask"].append(out["attention_mask"])
        results["labels"].append(out["labels"])
    return results

# Wrap partial so we can pass the tokenizer
map_func = partial(instruction_map_fn, tokenizer=tokenizer, task_type=task_type)

# Truncate training dataset if needed
if isinstance(train_samples, int) and train_samples > 0:
    raw_train_dataset = raw_train_dataset.select(range(train_samples))
elif train_samples == "all":
    pass  # Keep the original dataset unchanged
else:
    raise ValueError("Invalid 'train_samples' value. Must be >0 or 'all'.")

# Truncate validation dataset if needed
if isinstance(valid_samples, int) and valid_samples > 0:
    raw_val_dataset = raw_val_dataset.select(range(valid_samples))
elif valid_samples == "all":
    pass  # Keep the original dataset unchanged
else:
    raise ValueError("Invalid 'valid_samples' value. Must be >0 or 'all'.")

# Remove irrelevant columns
columns_to_remove = ["model_id", "revision_id", "id", "num_unique_activities"]
train_ds = raw_train_dataset.map(map_func, batched=True, remove_columns=columns_to_remove)
val_ds   = raw_val_dataset.map(map_func,   batched=True, remove_columns=columns_to_remove)

#-------------------------CUSTOM DATA COLLATOR-------------------------
def causal_data_collator(features, tokenizer):
    """
    Pads input_ids, attention_mask, and labels to the same length across the batch.
    We pad prompt tokens with -100 for labels so they're ignored in the loss.
    """
    batch_input_ids = [f["input_ids"] for f in features]
    batch_attention = [f["attention_mask"] for f in features]
    batch_labels = [f["labels"] for f in features]
    padded = tokenizer.pad(
        {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention
        },
        padding=True,
        return_tensors="pt",
    )
    
    max_length = padded["input_ids"].size(1)
    padded_labels = []
    for lbl in batch_labels:
        num_to_pad = max_length - len(lbl)
        padded_labels.append(lbl + [-100] * num_to_pad)
    padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return padded

def my_collator(features):
    return causal_data_collator(features, tokenizer)


#-------------------------CUSTOM METRICS-------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

#-------------------------CUSTOM CALLBACKS-------------------------
"""
Define a custom callback to rename checkpoint directories during training.
"""
class CustomCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        custom_name = f"{args.output_dir}/{save_dir}_step-{state.global_step}"
        if os.path.exists(checkpoint_dir):
            os.rename(checkpoint_dir, custom_name)
            print(f"Renamed checkpoint: {checkpoint_dir} -> {custom_name}")
        with open(os.path.join(args.output_dir, "latest_checkpoint.txt"), "w") as f:
            f.write(custom_name)

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss")
            grad_norm = logs.get("grad_norm")
            lr = logs.get("learning_rate")
            epoch = logs.get("epoch")
            
            if loss is not None and grad_norm is not None and lr is not None and epoch is not None:
                print(f"{{'loss': {loss:.4f}, 'grad_norm': {grad_norm}, 'learning_rate': {lr}, 'epoch': {epoch:.2f}}}")

#-------------------------TRAINING ARGUMENTS-------------------------
"""
Configure training arguments and initialize the trainer with the model, datasets, and data collator.
"""
training_args = TrainingArguments(
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    optim=optimizer,
    weight_decay=0.01,
    lr_scheduler_type = "linear",
    seed=3407,
    logging_steps=log_steps,
    output_dir="outputs",
    report_to="wandb",
    per_device_eval_batch_size = eval_batch_size,
    eval_strategy = IntervalStrategy.STEPS,
    eval_steps = eval_steps,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=save_steps,
    save_total_limit=5, 
)

#------------------------- INIT WANDB RUN -------------------------
wandb.init(
    project="fine-tune-model",
    name=f"{label_model_name}_{task_name}",
    config=training_args.to_dict(),
)

# ------------------------- GPU MEMORY -------------------------
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#------------------------- SHORT RUN (THROUGHPUT ESTIMATION) -------------------------
if short_run == True:
    short_run_args = copy.deepcopy(training_args)
    short_run_args.num_train_epochs = 0
    short_run_args.max_steps = 100
    short_run_args.eval_steps = 0
    short_run_args.eval_strategy = "no"
    short_run_args.save_strategy = "no"

    short_run_trainer = SFTTrainer(
        model=model,
        args=short_run_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=my_collator,
    )

    print("Measuring throughput on 100 steps...")
    start_time = time.time()
    short_run_trainer.train()
    warmup_time = time.time() - start_time
    print(f"Time for 100 steps: {warmup_time:.2f} seconds")

    steps_per_epoch = len(train_ds) // (train_batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs
    estimated_total_time_seconds = (warmup_time / 100) * total_steps
    print(f"Estimated total fine-tuning time: {format_time(estimated_total_time_seconds)}")

    wandb.log({"estimated_total_time_minutes": estimated_total_time_seconds / 60})

#-------------------------TRAIN THE MODEL--------------------------------------------
full_run_args = copy.deepcopy(training_args)
full_run_args.max_steps = -1
full_run_args.num_train_epochs = epochs

# Initialize and run the trainer
trainer = SFTTrainer(
    model=model,
    args=full_run_args,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=my_collator,
)

# Add the custom checkpoint callback
trainer.add_callback(CustomCheckpointCallback())
trainer.add_callback(CustomLoggingCallback())
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

# Print dataset statistics
print_stats("Train", raw_train_dataset)
print_stats("Validation", raw_val_dataset)
print_stats("Test", raw_test_dataset)

# Print Log steps
print(f"Log steps: {log_steps}")

# Print dynamic eval_steps and save_steps
print(f"Dynamic eval_steps={eval_steps}. (We have ~{num_updates_per_epoch} training steps per epoch.)")
print(f"Dynamic save_steps={save_steps}. (Checkpoints will be saved every {save_steps} steps.)")

# Scan the output directory for the most recent checkpoint
latest_checkpoint = None
trainer_stats = None
if os.path.exists("outputs"):
    checkpoints = [d for d in os.listdir("outputs") if "step-" in d]
    if checkpoints:
        latest_checkpoint = os.path.join("outputs", sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])

if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    trainer_stats = trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("No checkpoint found, starting training from scratch.")
    trainer_stats = trainer.train()

#------------------------- SAVE THE MODEL -------------------------
# Save the fine-tuned model and tokenizer.
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
wandb.save(save_dir)
print(f"Model and tokenizer saved to {save_dir}")
wandb.finish()

#------------------------- SHOW FINAL MEMORY AND TIME STATS -------------------------
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")