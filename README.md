---
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: DistilBERT-yelp-sentiment-analysis
  results: []
widget:
- text: "This restaurant has the best food"
  example_title: "Positive review"
- text: "This restaurant has the worst food"
  example_title: "Negative review"
datasets:
- noahnsimbe/yelp-dataset
# model-index:
#   - name: DistilBERT-yelp-sentiment-analysis
#     results:
#       - task:
#           type: text-classification
#           name: Restaurant reviews classification
#         dataset:
#           name: Yelp reviews dataset
#           type: noahnsimbe/yelp-dataset
#           split: eval
#         metrics:
#           - name: Accuracy
#             type: accuracy
#             value: 0.8930
#           - name: F1
#             type: f1
#             value: 0.7863
#             config: macro
#           - name: Recall
#             type: recall
#             value: 0.7768
#             config: macro
#           - name: Loss
#             type: loss
#             value: 0.2995
#             config: macro
#           - name: Precision
#             type: precision
#             value: 0.7976
#             config: macro
---


<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# DistilBERT-yelp-sentiment-analysis

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on [yelp-dataset dataset](https://huggingface.co/datasets/noahnsimbe/yelp-dataset).
It achieves the following results on the evaluation set:
- Loss: 0.2995
- Accuracy: 0.8930
- F1: 0.7863
- Precision: 0.7976
- Recall: 0.7768

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1     | Precision | Recall |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:------:|:---------:|:------:|
| 0.2996        | 1.0   | 9645  | 0.2995          | 0.8930   | 0.7863 | 0.7976    | 0.7768 |
| 0.2233        | 2.0   | 19290 | 0.3381          | 0.8966   | 0.7957 | 0.8015    | 0.7907 |


### Framework versions

- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2
