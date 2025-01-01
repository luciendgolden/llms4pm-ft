from unsloth import FastLanguageModel
import torch
from functools import partial
from trl import SFTTrainer
from transformers import TrainingArguments
from llama_fine_tuning_util import load_trace_data
import datetime
from transformers import TrainerCallback
import os

#--------------------------INITIALIZE VARIABLES-----------------------------------------------------------
max_seq_length = 5 # Choose any! We auto support RoPE Scaling internally!
dtype = "bfloat16" # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
label_model_name = model_name.replace("/", "_")
task_name = "T-SAD"
# train_samples = 1400
train_samples = "all" # for full dataset
# valid_samples = 20
valid_samples = "all" # for full dataset
epochs = 3
learning_rate = 1e-5
train_batch_size = 2
train_accum_steps = 16
eval_batch_size = 4
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Current timestamp
log_steps = 50
eval_save_steps = 500

save_dir = (f"{label_model_name}_{task_name}_samples-{train_samples}_epochs-{epochs}_"
            f"lr-{learning_rate}_batch-{train_batch_size}x{train_accum_steps}_time-{timestamp}")


#---------------------------INITIALIZE MODEL AND TOKENIZER--------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "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", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = True, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

#-------------------------LOAD THE DATA------------------------------------------------------------------------
raw_train_dataset, raw_val_dataset, _ = load_trace_data()

#---------------------------------BUILD THE PROMPT AND TOKENIZE INPUT-----------------------------------------
def build_instruction_encoding(example, tokenizer, max_length=512):
    """
    Returns a dict with input_ids, attention_mask, and labels
    so that the prompt is masked out (label = -100)
    and only 'True' or 'False' tokens are labeled.
    """
    # Convert label to "True" or "False"
    label_str = str(example["ds_labels"])

    # Build the prompt
    prompt_text = f"Set of process activities: {set(example['unique_activities'])}\nTrace: {example['trace']}\nValid: "

    # We can optionally add an EOS token at the end of the response:
    # answer_text = label_str + tokenizer.eos_token
    answer_text = label_str  # If you prefer to omit EOS, do it here

    # Tokenize prompt separately (no special tokens, so they do not get BOS or EOS in the middle)
    enc_prompt = tokenizer(prompt_text, add_special_tokens=False)
    # Tokenize the response
    enc_response = tokenizer(answer_text, add_special_tokens=False)
    
    # Combine
    input_ids = enc_prompt["input_ids"] + enc_response["input_ids"]
    attention_mask = enc_prompt["attention_mask"] + enc_response["attention_mask"]

    # Build labels: 
    #   - Prompt portion => -100 
    #   - Response portion => actual IDs
    labels = [-100] * len(enc_prompt["input_ids"]) + enc_response["input_ids"]

    # Truncate if needed
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def instruction_map_fn(examples, tokenizer):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    for i in range(len(examples["ds_labels"])):
        example = {
            "ds_labels": examples["ds_labels"][i],
            "unique_activities": examples["unique_activities"][i],
            "trace": examples["trace"][i],
        }
        out = build_instruction_encoding(example, tokenizer)
        results["input_ids"].append(out["input_ids"])
        results["attention_mask"].append(out["attention_mask"])
        results["labels"].append(out["labels"])
    return results

# Wrap partial so we can pass the tokenizer
map_func = partial(instruction_map_fn, tokenizer=tokenizer)

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
columns_to_remove = [
    "model_id", "revision_id", "id", "num_unique_activities"
]

train_ds = raw_train_dataset.map(map_func, batched=True, remove_columns=columns_to_remove)
val_ds   = raw_val_dataset.map(map_func,   batched=True, remove_columns=columns_to_remove)

# to check if tokenized correctly
# print(tokenizer.decode(train_ds[0]["input_ids"]))


#---------------------------------------ENSURE EQUAL INPUT LENGTH PER BATCH------------------------
def causal_data_collator(features, tokenizer):
    """
    Pads input_ids, attention_mask, and labels to the same length across the batch.
    We pad prompt tokens with -100 for labels so they're ignored in the loss.
    """
    # Separate out each field
    batch_input_ids = [f["input_ids"] for f in features]
    batch_attention = [f["attention_mask"] for f in features]
    batch_labels    = [f["labels"] for f in features]  # each is a list of ints
    
    # Let HF pad the inputs & attention mask
    padded = tokenizer.pad(
        {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention
        },
        padding=True,  # True => pad to the longest sequence in this batch
        return_tensors="pt",
    )
    
    # Now pad labels to the same sequence length
    max_length = padded["input_ids"].size(1)
    padded_labels = []
    for lbl in batch_labels:
        # pad up to max_length
        num_to_pad = max_length - len(lbl)
        padded_labels.append(lbl + [-100] * num_to_pad)
    
    padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    
    return padded

def my_collator(features):
    return causal_data_collator(features, tokenizer)


#----------------------------------CUSTOM LOGIC FOR RENAMING CHECKPOINTS------------------------
class CustomCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # Get the most recent checkpoint path
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Create your custom checkpoint name
        custom_name = f"{args.output_dir}/{save_dir}_step-{state.global_step}"
        
        # Rename the checkpoint directory
        if os.path.exists(checkpoint_dir):
            os.rename(checkpoint_dir, custom_name)
            print(f"Renamed checkpoint: {checkpoint_dir} -> {custom_name}")


#-----------------------------TRAINER SETUP-----------------------------------------------
training_args = TrainingArguments(
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=train_accum_steps,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    bf16 = True,
    fp16=False,
    logging_steps=log_steps,
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=3407,
    output_dir="outputs",
    report_to="none",
    per_device_eval_batch_size = eval_batch_size,
    eval_strategy = "steps",
    eval_steps = eval_save_steps,
    save_strategy="steps",
    save_steps=eval_save_steps,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=my_collator,

)

trainer.add_callback(CustomCheckpointCallback())

#-----------------------------------------START FINETUNING------------------------------
trainer.train()

#----------------------------------------LOCALLY SAVE THE LAST CHECKPOINT-----------------
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)