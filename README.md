# llms4pm-ft

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
