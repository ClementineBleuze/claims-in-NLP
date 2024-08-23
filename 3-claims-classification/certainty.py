import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils.utils import get_predictions_from_logits
from certainty_estimator.predict_certainty import CertaintyEstimator
from datasets import Dataset
import argparse

def get_iter_n():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_n", type = int, required = True)
    args = parser.parse_args()

    iter_n = args.iter_n
    return iter_n


def certainty_inference(iter_n):

    # load the estimators
    sentence_estimator = CertaintyEstimator('sentence-level', cuda = True)
    print("Successfully loaded sentence-level certainty estimator !")
    
    aspect_estimator = CertaintyEstimator('aspect-level', cuda = True)
    print("Successfully loaded aspect-level certainty estimator !")

    total = pd.read_csv("data/sentences-corr.csv")
    batch_size = 832711

    print(f"Processing indexes {12519965 + batch_size * iter_n }-{min(12519965 + batch_size * (iter_n+1), 15850807)}")
    total = total.loc[range(12519965 + batch_size * iter_n , min(12519965 + batch_size * (iter_n+1), 15850807))]
    print(total.shape)

    for i in tqdm(range(12519965 + batch_size * iter_n , min(12519965 + batch_size * (iter_n+1), 15850807))):
        sent = total.at[i, "sentence"]
        
        sent_certainty = sentence_estimator.predict(sent)[0]
        total.at[i, "sentence_certainty"] = sent_certainty
    
        ASPECTS = ["Number", "Extent", "Probability", "Suggestion", "Framing", "Condition"]
    
        aspect_certainty = aspect_estimator.predict(sent)[0]
        aspect_certainty = [[elt[0], elt[1]] for elt in aspect_certainty]
        aspect_dict = {elt[0]: elt[1] for elt in aspect_certainty}
        total.at[i, "aspect_certainty"] = str(aspect_certainty)
    
    for aspect in ASPECTS:
        if aspect in aspect_dict.keys():
            total.at[i, aspect] = aspect_dict[aspect]
        else:
            total.at[i, aspect] = "Absent"

    total.to_csv(f"data/sentences-corr-{iter_n}.csv", index = False)

if __name__ == "__main__":
    
    iter_n = get_iter_n()
    certainty_inference(iter_n)
    
    