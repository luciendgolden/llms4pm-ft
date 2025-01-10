"""
Fine-tunes a pre-trained language model (CLM) such as GPT-2, GPT-3, or Meta-Llama on custom tasks:
 - T-SAD  (Trace anomaly detection)
 - A-SAD  (Activity out-of-order detection)
 - S-NAP  (Semantic next-activity prediction)

Usage:
    python llama_hf_script.py <TASK_NAME>
    
Example:
    python llama_hf_script.py S-NAP

Task Description:
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

Notes:
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
from functools import partial

import torch
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
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
from unsloth import FastLanguageModel, is_bfloat16_supported

from prompt_tokenize import instruction_map_fn
from llama_fine_tuning_util import load_dataset, print_stats, format_time

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
def setup_environment():
    """
    Configure environment for minimal logs and faster GPU usage.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    torch.backends.cudnn.benchmark = True


def print_gpu_info():
    """
    Logs information about the current GPU device.
    """
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Training might be very slow.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)
    print(f"Using GPU: {gpu_name}, capability={gpu_capability}")
    print(torch.cuda.memory_summary())


# -------------------------------------------------------------------
# Constants / Task Mappings
# -------------------------------------------------------------------
TASK_TO_DATASET = {
    "T-SAD": ("data/T_SAD.csv", "TRACE_ANOMALY"),
    "A-SAD": ("data/A_SAD.csv", "OUT_OF_ORDER"),
    "S-NAP": ("data/S_NAP.csv", "NEXT_ACTIVITY"),
}


# -------------------------------------------------------------------
# Additional Helper Functions
# -------------------------------------------------------------------
def select_and_tokenize_dataset(dataset, n_samples, map_func, columns_to_remove):
    """
    Optionally select a subset of `n_samples` from `dataset`,
    then apply `map_func` to tokenize and remove unnecessary columns.
    """
    if isinstance(n_samples, int) and n_samples > 0:
        dataset = dataset.select(range(n_samples))
    elif n_samples == "all":
        pass
    else:
        raise ValueError(f"Invalid n_samples: {n_samples} (must be int>0 or 'all')")
    
    return dataset.map(map_func, batched=True, remove_columns=columns_to_remove)


def dynamic_eval_steps(num_updates_per_epoch):
    """
    Decide how often to run eval within each epoch.
    """
    MIN_EVAL_STEPS = 50
    MAX_EVAL_STEPS = 1000
    # Simple logic example
    desired_evals_per_epoch = 2
    if num_updates_per_epoch >= 10_000:
        desired_evals_per_epoch = 6
    elif num_updates_per_epoch >= 100_000:
        desired_evals_per_epoch = 12

    steps = max(1, num_updates_per_epoch // desired_evals_per_epoch)
    steps = max(MIN_EVAL_STEPS, min(steps, MAX_EVAL_STEPS))
    return steps


def dynamic_save_steps(num_updates_per_epoch, eval_steps, train_size):
    """
    Decide how often to save checkpoints based on the dataset size,
    combining logic with eval_steps for consistency.
    """
    MIN_SAVE_STEPS = 50
    MAX_SAVE_STEPS = 1000

    if train_size < 10_000:
        return eval_steps * 2
    elif 10_000 <= train_size <= 100_000:
        return eval_steps
    else:
        half = max(MIN_SAVE_STEPS, eval_steps // 2)
        return min(half, MAX_SAVE_STEPS)


def show_gpu_memory_usage():
    """
    Logs GPU memory usage at script start. 
    """
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}, max={max_memory}GB, reserved={start_memory}GB")


def run_short_test(model, tokenizer, train_dataset, val_dataset, training_args):
    """
    Runs a short run for throughput estimation on 100 steps, with no real training goal.
    """
    print("Running a short run for throughput estimation...")

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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(causal_data_collator, tokenizer=tokenizer),
    )

    start_time = time.time()
    short_run_trainer.train()
    elapsed = time.time() - start_time
    print(f"Time for 100 steps: {elapsed:.2f} seconds")

    steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.num_train_epochs
    est_seconds = (elapsed / 100) * total_steps
    print(f"Estimated total fine-tuning time: {format_time(est_seconds)}")


def find_latest_checkpoint(output_dir):
    """
    Finds the most recent checkpoint in `output_dir`.
    Returns the path or None if none is found.
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if "step-" in d]
    if not checkpoints:
        return None
    return os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])


def show_final_stats(trainer_stats):
    """
    Prints time and memory usage from the trainer stats object.
    """
    if not trainer_stats:
        return
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    runtime = trainer_stats.metrics.get('train_runtime', 0.0)
    print(f"{runtime} seconds used for training.")
    print(f"{round(runtime/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")


# -------------------------------------------------------------------
# Collator & Custom Callbacks
# -------------------------------------------------------------------
def causal_data_collator(features, tokenizer):
    """
    Pads input_ids, attention_mask, and labels to the same length across the batch.
    We pad prompt tokens with -100 for labels so they're ignored in the cross-entropy loss.
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
        padded_labels.append(lbl + ([-100] * num_to_pad))

    padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return padded


class CustomCheckpointCallback(TrainerCallback):
    """
    Renames checkpoint directories for clarity during training.
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        custom_name = f"{args.output_dir}/{self.save_dir}_step-{state.global_step}"
        if os.path.exists(checkpoint_dir):
            os.rename(checkpoint_dir, custom_name)
            print(f"Renamed checkpoint: {checkpoint_dir} -> {custom_name}")
        with open(os.path.join(args.output_dir, "latest_checkpoint.txt"), "w") as f:
            f.write(custom_name)


class CustomLoggingCallback(TrainerCallback):
    """
    Logs custom information, such as loss/grad-norm, at each log step.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            grad_norm = logs.get("grad_norm")
            lr = logs.get("learning_rate")
            epoch = logs.get("epoch")

            if all(v is not None for v in [loss, grad_norm, lr, epoch]):
                print(f"{{'loss': {loss:.4f}, 'grad_norm': {grad_norm}, "
                      f"'learning_rate': {lr}, 'epoch': {epoch:.2f}}}")


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    """
    Main entry point for fine-tuning a Llama-based model on T-SAD, A-SAD, or S-NAP tasks.
    """
    setup_environment()
    print_gpu_info()

    # ------------------------------ 
    # Parse task from command line
    # ------------------------------
    if len(sys.argv) < 2:
        print("Usage: python llama_hf_script.py <TASK_NAME>")
        sys.exit(1)
    task_name = sys.argv[1]

    if task_name not in TASK_TO_DATASET:
        raise ValueError(f"Unsupported task: {task_name}. Choose from {list(TASK_TO_DATASET.keys())}")

    dataset_file, task_type = TASK_TO_DATASET[task_name]

    # ------------------------------ 
    # Load dataset
    # ------------------------------
    # TODO: Adjust fraction for faster testing
    fraction = 0.01
    train_ds_raw, val_ds_raw, test_ds_raw = load_dataset(dataset_file, task_type, fraction)
    train_size = len(train_ds_raw)
    print(f"Train dataset size: {train_size}, Validation={len(val_ds_raw)}, Test={len(test_ds_raw)}")

    # Save test set to CSV for later re-use
    test_csv_path = f"test_set_{task_name}.csv"
    test_ds_raw.to_pandas().to_csv(test_csv_path, index=False)
    print(f"Test set saved to {test_csv_path}")

    # ------------------------------ 
    # Model & Training Config
    # ------------------------------
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    label_model_name = model_name.replace("/", "_")

    max_seq_length = 512
    train_batch_size = 2
    grad_acc_steps = 16
    num_epochs = 3
    lr = 1e-5
    eval_batch_size = 16
    use_4bit_quant = False

    short_run = False
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Name your final output checkpoint directory
    output_checkpoint_dir = (f"{label_model_name}_{task_name}_"
                             f"epochs-{num_epochs}_lr-{lr}_"
                             f"batch-{train_batch_size}x{grad_acc_steps}_"
                             f"time-{timestamp}")

    # ------------------------------ 
    # Prepare Model & Tokenizer
    # ------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None if not is_bfloat16_supported() else torch.bfloat16,
        load_in_4bit=use_4bit_quant,
        token="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",
    )
    model = FastLanguageModel.get_peft_model(
        base_model=model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,  # for memory efficiency if needed
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # ------------------------------ 
    # Tokenize & Prepare Datasets
    # ------------------------------
    map_func = partial(instruction_map_fn, tokenizer=tokenizer, task_type=task_type)

    # Decide how many train/val samples to use: "all" => entire dataset
    train_samples = "all"
    val_samples = "all"

    train_ds = select_and_tokenize_dataset(
        dataset=train_ds_raw,
        n_samples=train_samples,
        map_func=map_func,
        columns_to_remove=["model_id", "revision_id", "id", "num_unique_activities"]
    )
    val_ds = select_and_tokenize_dataset(
        dataset=val_ds_raw,
        n_samples=val_samples,
        map_func=map_func,
        columns_to_remove=["model_id", "revision_id", "id", "num_unique_activities"]
    )

    # ------------------------------ 
    # Training Arguments
    # ------------------------------
    num_updates_per_epoch = len(train_ds_raw) // (train_batch_size * grad_acc_steps)
    eval_steps = dynamic_eval_steps(num_updates_per_epoch)
    save_steps = dynamic_save_steps(num_updates_per_epoch, eval_steps, train_size)

    training_args = TrainingArguments(
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        fp16=(not is_bfloat16_supported()),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        logging_steps=50,
        output_dir="outputs",
        report_to="wandb",
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=eval_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=save_steps,
        save_total_limit=5,
    )

    # ------------------------------ 
    # Init Weights & Biases
    # ------------------------------
    wandb.init(
        project="fine-tune-model",
        name=f"{label_model_name}_{task_name}",
        config=training_args.to_dict(),
    )

    # Show GPU memory usage
    show_gpu_memory_usage()

    # If short_run is True, do a quick throughput test on 100 steps
    if short_run:
        run_short_test(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            training_args=training_args
        )

    # ------------------------------ 
    # SFTTrainer Setup
    # ------------------------------
    full_run_args = copy.deepcopy(training_args)
    full_run_args.max_steps = -1
    full_run_args.num_train_epochs = num_epochs

    trainer = SFTTrainer(
        model=model,
        args=full_run_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(causal_data_collator, tokenizer=tokenizer),
    )

    trainer.add_callback(CustomCheckpointCallback(output_checkpoint_dir))
    trainer.add_callback(CustomLoggingCallback())

    # ------------------------------ 
    # Print Dataset Stats
    # ------------------------------
    print_stats("Train", train_ds_raw)
    print_stats("Validation", val_ds_raw)
    print_stats("Test", test_ds_raw)

    # Log dynamic steps
    print(f"Dynamic eval_steps={eval_steps} (We have ~{num_updates_per_epoch} steps per epoch).")
    print(f"Dynamic save_steps={save_steps} (Checkpoints saved every {save_steps} steps).")

    # Possibly resume from a previous checkpoint
    latest_ckpt = find_latest_checkpoint("outputs")
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        trainer_stats = trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer_stats = trainer.train()

    # Save final model
    model.save_pretrained(output_checkpoint_dir)
    tokenizer.save_pretrained(output_checkpoint_dir)
    wandb.save(output_checkpoint_dir)
    print(f"Model and tokenizer saved to {output_checkpoint_dir}")
    wandb.finish()

    # Print final time/memory usage
    show_final_stats(trainer_stats)



if __name__ == "__main__":
    main()