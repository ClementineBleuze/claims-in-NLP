import numpy as np
import pickle
import torch
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import spacy
import sys
from utils.Corpus import Corpus

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from datasets import Dataset
from typing import List

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def get_dataset_input(ds: Dataset, path_to_data:str, type = "text")->List:
    """A function that takes a Dataset as parameter and that returns a list of inputs for a classification model. Inputs can be formatted in different ways:
    - 'text' : simply the text of the sentence to classify, e.g 'This is my sentence.'
    - 'prefix_text': the sentence prefixed with the name of the paper section, e.g 'section: Introduction, text: This is my sentence.'
    - 'prefix_SEP':  the sentence prefixed with the name of the paper section using a special separator, e.g 'Introduction [SEC] This is my sentence.'
    - 'prefix_cont_lr_SEP': the sentence prefixed with the name of the paper section and surrounded by its left and right context, using special separators, e.g 'Introduction [SEC] Left context [SEP] This is my sentence [SEP] Right context [SEP]'
    - 'prefix_cont_ll_SEP': the sentence prefixed with the name of the paper section and preceded by its 2 preceding sentences, using special separators, e.g 'Introduction [SEC] Left context [SEP] Left context [SEP] This is my sentence [SEP]'
    
    """
    input = None
    
    if type == "text":
        input = ds["text"]

    elif type == "prefix_text":
        input = []
        for text, sec in zip(ds["text"], ds["section"]):
            input.append(f"section: {sec}, text: {text}")

    elif type == "prefix_SEP":
        input = []
        for text, sec in zip(ds["text"], ds["section"]):
            input.append(f"{sec}[SEC]{text}")

    elif type == "prefix_cont_lr_SEP":
        df = pd.read_csv(path_to_data)
        input = []
        for text, sec, l_id, r_id in zip(ds["text"], ds["section"], ds["-1"], ds["+1"]):
            ls = df[df["id"] == l_id].text.values[0] if l_id != -1 else ""
            rs = df[df["id"] == r_id].text.values[0] if r_id != -1 else ""
            input.append(f"{sec}[SEC]{ls}[SEP]{text}[SEP]{rs}")

    elif type == "prefix_cont_ll_SEP":
        input = []
        df = pd.read_csv(path_to_data)
        for text, sec, ll_id, l_id in zip(ds["text"], ds["section"], ds["-2"], ds["-1"]):
            lls = df[df["id"] == ll_id].text.values[0] if ll_id != -1 else ""
            ls = df[df["id"] == l_id].text.values[0] if l_id != -1 else ""
            input.append(f"{sec}[SEC]{lls}[SEP]{ls}[SEP]{text}")

    return input

def get_dataset_label(ds: Dataset, path_to_data, LABELS, type = "1-hot")->List:
    label = None
    if type == "1-hot":
        label = ds["label"]

    elif type == "1-hot_cont_lr":
        df = pd.read_csv(path_to_data)
        label = []
        for lab, l_id, r_id in zip(ds["label"], ds["-1"], ds["+1"]):
            l = []
            l.extend(json.loads(df[df["id"] == l_id].label_as_one_hot.values[0]) if l_id != -1 else [0 for _ in range(len(LABELS))])
            l.extend(lab)
            l.extend(json.loads(df[df["id"] == r_id].label_as_one_hot.values[0]) if r_id != -1 else [0 for _ in range(len(LABELS))])
            label.append(l)

    elif type == "1-hot_cont_ll":
        df = pd.read_csv(path_to_data)
        label = []
        for lab, ll_id, l_id in zip(ds["label"], ds["-2"], ds["-1"]):
            l = []
            l.extend(json.loads(df[df["id"] == ll_id].label_as_one_hot.values[0]) if ll_id != -1 else [0 for _ in range(len(LABELS))])
            l.extend(json.loads(df[df["id"] == l_id].label_as_one_hot.values[0]) if l_id != -1 else [0 for _ in range(len(LABELS))])
            l.extend(lab)
            label.append(l)
        
    return label

# how many parameters in the model ?
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def startswith_in_list(s, l):
    for elt in l:
        if s.startswith(elt):
            return True
    return False

def get_predictions_from_logits(logits, strategy = "", threshold = 0.5, use_sigmoid = True):

    # if needed, start by applying the sigmoid function to logits, to have values between 0 and 1
    if use_sigmoid:
        sigmoid = torch.nn.Sigmoid()
        probas = sigmoid(torch.tensor(logits))
    else:
        probas = torch.tensor(logits)

    if strategy == "argmax":
        predictions = np.zeros(probas.shape)
        for i, row in enumerate(probas):
            max_ = max(row).item()
            predictions[i][np.where(row == max_)] = 1

    elif strategy == "constraints":
        predictions = np.zeros(probas.shape)
        for i, row in enumerate(probas):
            argmax = np.argmax(row)
            predictions[i][argmax] = 1

            if argmax != 7:
                other_labels = [j for j in range(8) if i!= argmax and row[j] >= threshold and row[j] > row[7]]
                predictions[i][other_labels] = 1

    else:
        predictions = np.zeros(probas.shape)
        predictions[np.where(probas >= threshold)] = 1

    return predictions

def radar_plot(df:pd.DataFrame):
       
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
     
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
     
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size = 10)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25,0.50,0.75, 1.00], ["0.25","0.50","0.75", "1.00"], color="grey", size=7)
    plt.ylim(0,1)
     
    
    # ------- PART 2: Add plots
    
    #palette = ["red", "blue", "green", "darkred", "darkblue", "darkgreen"]
    for i in range(df.shape[0]):
        values = df.loc[i].drop('model').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label= df.at[i, "model"])
        
     
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Show the graph
    plt.show()

# Custom JSON encoder
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Corpus):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)

def load_corpus_object(path:str)->Corpus:

    # open the json file to get the str/dict object
    with open(path, "r") as f:
        corpus_s = json.load(f)

    # convert it to a dict object if needed
    if type(corpus_s) == str:
        corpus_dict = json.loads(corpus_s)
    else:
        corpus_dict = corpus_s

    # finally, convert it back to a Corpus object
    corpus = Corpus.from_dict(corpus_dict)

    for paper in corpus.papers:
        paper.corpus = corpus
    for paper in corpus.papers_with_errors:
        paper.corpus = corpus

    return corpus
    
