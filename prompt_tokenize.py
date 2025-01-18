"""
Define functions to prepare the training data by creating prompts and tokenizing inputs/labels.
"""
def build_instruction_encoding(example, tokenizer, max_length=512, task_type="TRACE_ANOMALY"):
    """
    Returns a dict with input_ids, attention_mask, and labels
    so that the prompt is masked out (label = -100)
    and only the final 'answer' tokens are labeled.
    """
    if task_type == "TRACE_ANOMALY":
        ds_label = example["ds_labels"]
        label_str = "True" if ds_label else "False"
        prompt_text = f"Set of process activities: {set(example['unique_activities'])}\nTrace: {example['trace']}\nValid: "
    elif task_type == "OUT_OF_ORDER":
        ds_label = example["ds_labels"]
        label_str = "True" if ds_label else "False"
        eventually = example["eventually_follows"]
        prompt_text = f"Set of process activities: {set(example['unique_activities'])}\n1. Activity: {eventually[0]}\n2. Activity: {eventually[1]}\nValid: "
    elif task_type == "NEXT_ACTIVITY":
        list_of_activities = sorted(set(list(example["unique_activities"])))
        prefix_str = example["prefix"]
        next_act   = example["next"]
        
        if "[END]" not in list_of_activities:
            list_of_activities.append("[END]")
        
        prompt_text = (
            f"List of activities: {list_of_activities}\n"
            f"Sequence of activities: {prefix_str}\n"
            "Next activity: "
        )
        
        label_str = next_act
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    enc_prompt = tokenizer(text=prompt_text, add_special_tokens=False)
    enc_answer = tokenizer(text=label_str, add_special_tokens=False)
    
    input_ids = enc_prompt["input_ids"] + enc_answer["input_ids"]
    attention_mask = enc_prompt["attention_mask"] + enc_answer["attention_mask"]
    labels = ([-100] * len(enc_prompt["input_ids"])) + enc_answer["input_ids"]
    
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def instruction_map_fn(examples, tokenizer, task_type):
    """
    Map function to tokenize inputs and labels for the training dataset.
    """
    records = []
    if task_type in ["TRACE_ANOMALY", "OUT_OF_ORDER"]:
        for i in range(len(examples["ds_labels"])):
            record = {
                "ds_labels": examples["ds_labels"][i],
                "unique_activities": examples["unique_activities"][i],
            }
            if task_type == "TRACE_ANOMALY":
                record["trace"] = examples["trace"][i]
            elif task_type == "OUT_OF_ORDER":
                record["eventually_follows"] = examples["eventually_follows"][i]

            records.append(record)
    elif task_type == "NEXT_ACTIVITY":
        for i in range(len(examples["prefix"])):
            record = {
                "prefix": examples["prefix"][i],
                "next": examples["next"][i],
                "unique_activities": examples["unique_activities"][i],
            }
            records.append(record)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    encodings = [
        build_instruction_encoding(
            record,
            tokenizer=tokenizer,
            max_length=512,
            task_type=task_type
        )
        for record in records
    ]
    
    input_ids = [encoding["input_ids"] for encoding in encodings]
    attention_mask = [encoding["attention_mask"] for encoding in encodings]
    labels = [encoding["labels"] for encoding in encodings]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }