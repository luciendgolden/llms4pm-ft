# llms4pm-ft

Re-Implementation of the LLM fine-tuning part of the paper [Evaluating the Ability of LLMs to Solve
Semantics-Aware Process Mining Tasks](https://arxiv.org/pdf/2407.02310)

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

```sh
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
