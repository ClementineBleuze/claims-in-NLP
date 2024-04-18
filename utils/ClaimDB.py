import pandas as pd
from utils.Corpus import Corpus
from typing import List
import json
import pickle
import tqdm
import random

class ClaimDB:
    """A class to represent a claim database. A claim database is a collection of corpora and a list of sentences found in these corpora.
    Among these (candidate) sentences, some are claims."""
    
    def __init__(self, corpora:List[Corpus] = [], from_pickle_file:str = "", annotated_idx_path:str = ""):
        
        if from_pickle_file != "":

            with open(from_pickle_file, "rb") as f:
                cdb = pickle.load(f)
            
            self.corpora = cdb.corpora
            self.candidates = cdb.candidates
            self.idx_map = cdb.idx_map
            self.claims = cdb.claims
            self.annotated_idx_path = cdb.annotated_idx_path
        
        else:

            self.corpora = corpora
            self.candidates, self.idx_map = self.init_candidates()
            self.claims = None
            self.annotated_idx_path = annotated_idx_path


    def init_candidates(self):

        data = []

        for corpus in tqdm.tqdm(self.corpora):
            for paper in tqdm.tqdm(corpus.papers):
                candidates = paper.content[paper.content["candidate"] == True]

                for i, candidate in candidates.iterrows():

                    data.append({
                        "corpus": corpus.name,
                        "paper_id": paper.id,
                        "sentence_id": candidate["id"],
                        "sentence": candidate["sentence"],
                        "section": candidate["section"],
                    })

        df = pd.DataFrame(data)
        df["idx"] = df.index
        df = df[["idx", "corpus", "paper_id", "sentence_id", "sentence", "section"]]
    
        idx_map = {idx: (row["corpus"], row["paper_id"], row["sentence_id"]) for idx, row in df.iterrows()}

        return df, idx_map
    
    def prepare_for_doccano_format(df:pd.DataFrame)-> pd.DataFrame:
        pass

    def draw_random_idx_from_corpus(self, corpus:str, n:int)->List[str]:
        # get all the possible indexes to choose from
        idx_list = list(self.candidates[self.candidates["corpus"] == corpus]["paper_id"].unique())

        # exclude those that have already been annotated
        with open(self.annotated_idx_path, "r") as f:
            annotated_idx = json.load(f)
        
        idx_list = list(set(idx_list).difference(set(annotated_idx)))

        # draw n random indexes
        return random.sample(idx_list, n)

