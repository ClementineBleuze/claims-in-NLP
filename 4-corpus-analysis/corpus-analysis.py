import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import json
import re
from tqdm import tqdm
import spacy
from collections import Counter

DONE = []
#DONE = ["context-AIC", "contribution-AIC", "result", "impact", "directions", "limitation", "outline-AIC"]

## LOAD DATAnlp = 
# labels
with open("data/labels.json") as f:
    LABELS = json.load(f)

# sentences
print("Loading sentences.csv...")
df = pd.read_csv("data/sentences.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns = ["Unnamed: 0"])
    
df["label_anno"] = df["label_anno"].apply(lambda x: json.loads(x.replace("'", '"')))
df["label_pred"] = df["label_pred"].apply(lambda x: json.loads(x.replace("'", '"')))
print("Done !")

## CREATE SUB-DATAFRAMES PER CATEGORY
# pure categories
df_categs = []
for label in LABELS:
    df_categ= df[df["label_pred"].apply(lambda x: x == [label])]
    df_categs.append((df_categ.index.tolist(), df_categ.sentence.values.tolist()))

## INIT SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# Remove '[' and ']' from prefixes and suffixes to avoid poor segmentation
prefixes = list(nlp.Defaults.prefixes)
prefixes.remove("\[")
prefix_regex = spacy.util.compile_prefix_regex(prefixes)
nlp.tokenizer.prefix_search = prefix_regex.search

suffixes = list(nlp.Defaults.suffixes)
suffixes.remove("\]")
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", 
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

DEPS = nlp.get_pipe("parser").labels
NER = nlp.get_pipe("ner").labels

columns = ["sentence_id", "#char", "#tok"] + POS_TAGS + list(DEPS) + list(NER)

for i, (idx_list, sent_list) in enumerate(df_categs):

    # idx_list = idx_list[:100]
    # sent_list = sent_list[:100]

    if LABELS[i] in DONE:
        print(f"Skipping category {LABELS[i]} because it has already been processed !")
        continue
        
    print(f"Now processing sentences from category {LABELS[i]} ({len(sent_list)})")

    # init vocab and data
    with open(f"data/analyses/voc-{LABELS[i]}.json", "r") as f:
        voc = json.load(f)

    data = []

    for idx, sent in tqdm(zip(idx_list, sent_list), total = len(sent_list)):
        
        if type(sent) !=str:
            sent = str(sent)

        if idx > 12519965:
            
            doc = nlp(sent)
            pos_tags = [t.pos_ for t in doc]
            deps = [t.dep_ for t in doc]
            ents = [e.label_ for e in doc.ents]
            lemmas =[t.lemma_ for t in doc]
        
            #voc.extend(lemmas)
            for lemma in lemmas:
                if lemma in voc.keys():
                    voc[lemma] += 1
                else:
                    voc[lemma] = 1
            
            l = [idx]
            
            # sentence length
            l.append(len(sent)) #characters
            l.append(len(doc)) # tokens
        
            # POS_TAGS
            for tag in POS_TAGS:
                l.append(len([p for p in pos_tags if p==tag]))
        
            # DEPS
            for dep in DEPS:
                l.append(len([d for d in deps if d==dep]))
        
            # NE 
            for ner in NER:
                l.append(len([e for e in ents if e==ner]))
        
            data.append(l)
    
    df_stats =  pd.DataFrame(data = data, columns = columns)
    df_stats.to_csv(f"data/analyses/stats-{LABELS[i]}-new.csv", index = False)

    #counter = Counter(voc)
    with open(f"data/analyses/voc-{LABELS[i]}-new.json", "w") as f:
        json.dump(voc, f)

        



