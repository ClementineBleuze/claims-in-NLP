import numpy as np
import torch
import pandas as pd
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import BatchAllTripletLoss
from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from utils.utils import get_dataset_input, get_dataset_label

# check that a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# connect to Huggingface
login(token = "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", add_to_git_credential = True)

# data loading
with open("data/train_final.pkl", "rb") as f:
    train_ds = pickle.load(f)

with open("data/eval_final.pkl", "rb") as f:
    eval_ds = pickle.load(f)

with open("data/test_final.pkl", "rb") as f:
    test_ds = pickle.load(f)

with open("data/labels.json", "r") as f:
    LABELS = json.load(f)

input_data = {"prefix_SEP": {},
              "prefix_cont_lr_SEP": {},
              "prefix_cont_ll_SEP": {},
             }
PATH_TO_DATA = "data/processed-annotations-11-06.csv"

for input_type in input_data.keys():
    for split, ds in zip(["train", "eval", "test"], [train_ds, eval_ds, test_ds]):
        input_data[input_type][split] = get_dataset_input(ds, PATH_TO_DATA, type = input_type)

# we just use the default multi-label config (encoded as 1-hot vectors)
y_train = get_dataset_label(train_ds)
y_eval = get_dataset_label(eval_ds)
y_test = get_dataset_label(test_ds)

labels = {"train": y_train, "eval": y_eval, "test": y_test}

# define tokenization function
def tokenize_function(tokenizer):
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    return tokenize
    
CHECKPOINTS = {
    "roberta": "FacebookAI/roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "scibert": "allenai/scibert_scivocab_uncased",
}

input_data_per_model_family = {}

for model_family in CHECKPOINTS.keys():
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS[model_family])
    print(len(tokenizer.vocab))
    #add new special token separator for section
    special_tokens_dict = {'additional_special_tokens': ['[SEC]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(len(tokenizer.vocab))
    
    input_data_per_model_family[model_family] = {}
    
    # apply on the data
    for input_type in input_data.keys():
        input_data_per_model_family[model_family][input_type] = {}
        for split in input_data[input_type].keys():
            dataset = Dataset.from_dict({"text": input_data[input_type][split], "label": labels[split]})
            input_data_per_model_family[model_family][input_type][split] = dataset.map(tokenize_function(tokenizer), batched = True)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_samples_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_weighted': f1_weighted_average,
               'f1_samples': f1_samples_average,
               'f1_macro': f1_macro_average,
               'f1_micro' : f1_micro_average,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


with open("results/BERT.pkl", "rb") as f:
    results = pickle.load(f)

for model_family in input_data_per_model_family.keys():
    if model_family not in results.keys():
        results[model_family] = {}
    for input_type in  input_data_per_model_family[model_family].keys():

        # skip models already trained
        if input_type in results[model_family].keys():
            continue

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # load the model to be fine-tuned
        model_name = f"{model_family}_{input_type}"
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINTS[model_family],
                                                          num_labels = len(LABELS),
                                                          problem_type = "multi_label_classification")

        # load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS[model_family])
        # add new special token separator for section
        special_tokens_dict = {'additional_special_tokens': ['[SEC]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        # resize model embeddings
        model.resize_token_embeddings(len(tokenizer))

        # datasets
        train_ds = input_data_per_model_family[model_family][input_type]["train"]
        eval_ds = input_data_per_model_family[model_family][input_type]["eval"]
        test_ds = input_data_per_model_family[model_family][input_type]["test"]

        # safety check: does model inference work correctly ?
        print(f"Successfully loaded model {model_name} !\n Small example :\n")
        ex_sent = "We believe that this will be useful for other languages too."
        print(f"Example sentence: {ex_sent}\n")
        inputs = tokenizer(ex_sent, return_tensors="pt")
        logits = model(**inputs).logits
        probas = torch.nn.Sigmoid()(logits.squeeze().cuda())
        print(probas)

        # define the training arguments
        training_args = TrainingArguments(
            output_dir=f"models/NEW/{model_name}",
            push_to_hub = True,
            num_train_epochs= 15,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.1,
            learning_rate = 1e-5,
            logging_dir="./logs",
            eval_strategy="steps",
            eval_steps = 500,
            save_strategy = "steps",
            save_steps = 500,
            logging_steps = 500,
            save_total_limit = 2,
            load_best_model_at_end = True,
            metric_for_best_model = "f1_weighted",
        )

        # define the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics = compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)] # stop when the performance degrades for 3 consecutive eval steps
        )

        # let the model run on the GPU
        model = model.to(device)

        # train
        trainer.train()
        trainer.evaluate()

        # push to hub
        trainer.push_to_hub()

        # re-run the initial example
        ex_sent = "We believe that this will be useful for other languages too."
        print(f"Example sentence: {ex_sent}\n")
        inputs = tokenizer(ex_sent, return_tensors="pt")
        logits = model(**inputs).logits
        probas = torch.nn.Sigmoid()(torch.tensor(logits))
        print(probas)

        # on the test set
        logits = trainer.predict(test_ds).predictions
        sigmoid = torch.nn.Sigmoid()
        probas = sigmoid(torch.tensor(logits))
        predictions = np.zeros(probas.shape)
        predictions[np.where(probas >= 0.50)] = 1
        gold = test_ds["label"]
        cr = classification_report(gold, predictions, target_names = LABELS, zero_division = 0, output_dict = True)

        # store results
        results[model_family][input_type] = {
            "log_history" : trainer.state.log_history,
            #"classification_report": cr,
        }

        # save the updated results
        with open("results/BERT.pkl", "wb") as f:
            pickle.dump(results, f)