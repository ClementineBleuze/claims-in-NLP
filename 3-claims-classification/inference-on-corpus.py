import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils.utils import get_predictions_from_logits
from datasets import Dataset
import argparse

def get_iter_n():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_n", type = int, required = True)
    args = parser.parse_args()

    iter_n = args.iter_n
    return iter_n

def tokenize_function(tokenizer):
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    return tokenize


def model_inference(iter_n):

    # load the dataset containing all of the sentences
    df = pd.read_csv("data/sentences-corr.csv")
    df = df[(df["label_pred"].apply(lambda x: x == "[]")) & (df["label_anno"].apply(lambda x: x == "[]"))]
    print(df.shape)
    print("Successfully loaded sentences-corr.csv !")
    
    # load the inputs
    with open("data/inference-inputs-for-left-out-sents.json", "r") as f:
        inputs = json.load(f)
    
    print("Successfully loaded inference-inputs-for-left-out-sents.json !")

    # list of corresponding indexes
    batch_size = 832710
    idx = [ip[0] for ip in inputs[iter_n * batch_size : min([(iter_n+1) * batch_size, len(inputs)])]]
    df = df.loc[idx]
    print(f"Now we only have {df.shape[0]} rows !")
        
    # connect to Huggingface
    login(token = "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", add_to_git_credential = True)
    
    # load labels
    with open("data/labels.json", "r") as f:
        LABELS = json.load(f)
        
    # load the model
    model_checkpoint = "ClementineBleuze/scibert_prefix_cont_ll_SEP"
    tokenizer_checkpoint = "allenai/scibert_scivocab_uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                               num_labels = len(LABELS),
                                                               problem_type = "multi_label_classification")
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    # add new special token separator for section
    special_tokens_dict = {'additional_special_tokens': ['[SEC]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # create a trainer
    args = TrainingArguments(output_dir = "results", per_device_eval_batch_size = 16)
    trainer = Trainer(model = model, args = args)

    test_ds = Dataset.from_dict({"text": [ip[1] for ip in inputs[iter_n * batch_size : min([(iter_n+1) * batch_size, len(inputs)])]]})
    tokenized_test_ds = test_ds.map(tokenize_function(tokenizer), batched = True)
    print("Tokenized data !")
    
    # data inference
    logits = trainer.predict(tokenized_test_ds).predictions
    predictions = get_predictions_from_logits(logits, strategy = "constraints", threshold = 0.5, use_sigmoid = True)
    print("Made all predictions !")
    
    # store the labels !
    
    for sentence_id, prediction in zip(idx, predictions):
        df.at[sentence_id, "label_pred"] = str(prediction)
    
    df.to_csv(f"data/sentences-corr-{iter_n}.csv", index = False)

if __name__ == "__main__":
    
    iter_n = get_iter_n()
    model_inference(iter_n)
    
    