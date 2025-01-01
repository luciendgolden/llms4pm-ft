from sklearn.metrics import f1_score
import pandas as pd
from unsloth import FastLanguageModel
from datasets import Value
from llama_fine_tuning_util import load_trace_data
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
from functools import partial

tqdm.pandas()  # Enable progress bar


max_seq_length = 5 # Choose any! We auto support RoPE Scaling internally!
dtype = "bfloat16" # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-3.1-8B-Instruct", # or the fine-tuned model locally
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "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"
    )

FastLanguageModel.for_inference(model)

def generate_binary_output(prompt) -> str:
    # if model_name == MISTRAL_MODEL:
    #     prompt = "[INST]" + prompt + "[/INST]"
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_tokens = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
    )
    decoded = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    decoded_text = decoded[0].strip().lower()
    # print('Prediction: ', decoded_text)
    if "true" in decoded_text:
        return True
    elif "false" in decoded_text:
        return False
    print(decoded[0], "neither Yes nor No")
    print(tokenizer.decode(output_tokens[0][0]))
    return False  


def compute_y(example):
    prompt = f"Set of process activities: {example['unique_activities']}\nTrace: {example['trace']}\nValid: "
    example["y"] = generate_binary_output(prompt)
    return example


def run_predictions_loop(val_ds):
    val_ds = val_ds.map(compute_y, desc="Generating outputs for validation")
    predicted_labels = val_ds['y']
    return predicted_labels

def evaluate(val_ds, predicted_labels):
    result_records = []
    true_labels = val_ds["ds_labels"]       
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    result_records.append(f1_macro)
        
    df = pd.DataFrame(result_records)
    return df

_, _, raw_test_dataset = load_trace_data()
# raw_test_dataset = raw_test_dataset.cast_column("ds_labels", Value("bool"))
# short_test_dataset = raw_test_dataset.select(range(2000))

labels = run_predictions_loop(raw_test_dataset)
results = evaluate(raw_test_dataset, labels)
print(results)