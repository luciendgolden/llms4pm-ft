# llms4pm-ft

Re-Implementation of the LLM fine-tuning part of the paper 

[Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks](https://arxiv.org/pdf/2407.02310).

## Fine-tuning
The fine-tuning experiments were re-implemented since struggling with the trident-bpm framework. The original implementation can be found here:

[Trident-bpm Repository](https://github.com/fdschmidt93/trident-bpm)

## Google Colab Notebook

A Google Colab notebook is provided to demonstrate how to fine-tune large language models for process mining tasks. The notebook includes step-by-step instructions and code snippets to guide users through the fine-tuning process.

The notebook can be accessed here:

[Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks Notebook](notebooks/Evaluating_the_Ability_of_LLMs_to_Solve_Semantics_Aware_Process_Mining_Tasks.ipynb)

## Summary of Final Hyperparameters

Here's how your hyperparameters align with the desired settings:    

| **Parameter**               | **Value for Llama & Mistral** | **Value for RoBERTa** |
|-----------------------------|-------------------------------|-----------------------|
| **Optimizer**               | AdamW                         | AdamW                 |
| **Initial Learning Rate**   | 1e-5                          | 1e-5                  |
| **Number of Runs**          | 3                             | 3                     |
| **Batch Size per Instance** | 2                             | 32                    |
| **Gradient Accumulation**   | 16                            | N/A                   |
| **Effective Batch Size**    | 32                            | 32                    |
| **Number of Epochs**        | 3                             | 10                    |

## Install

Install the required packages on-premise.

```sh
pip install -r requirements.txt
```

Install the required packages for Google Colab.

```sh
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```