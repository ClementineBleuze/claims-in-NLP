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
                        "year" : paper.year,
                        "sentence_id": candidate["id"],
                        "sentence": candidate["sentence"],
                        "section": candidate["section"],
                    })

        df = pd.DataFrame(data)
        df["idx"] = df.index
        df = df[["idx", "corpus", "paper_id", "year", "sentence_id", "sentence", "section"]]
    
        idx2coord = {idx: (row["corpus"], row["paper_id"], row["sentence_id"]) for idx, row in df.iterrows()}

        return df, idx2coord

    def get_candidate_by_id(self, id:int):
        return self.candidates.loc[id]
    
    def get_candidates_by_paper_id(self, paper_id:str, corpus_name:str):
        coord2idx = {v:k for k,v in self.idx_map.items()}

        # get the paper
        c = self.get_corpus_by_name(corpus_name)
        p = c.get_paper_by_id(paper_id)

        paper_cands = p.content[p.content["candidate"] == True]
        paper_cands_idx = [coord2idx[(corpus_name, paper_id, i)] for i in paper_cands.index]

        return self.candidates.loc[paper_cands_idx]
        
    
    def get_corpus_by_name(self, name:str)->Corpus:
        for corpus in self.corpora:
            if corpus.name == name:
                return corpus
        return None
    
    def prepare_for_doccano_format(self, df:pd.DataFrame)-> pd.DataFrame:
        """A function to prepare a dataframe of sentences for Doccano format
        - df : a pandas DataFrame with columns {corpus, paper_id, sentence_id, sentence, section}"""

        data = []
        coord2idx = {v:k for k,v in self.idx_map.items()}


        for i, row in df.iterrows():
            c = self.get_corpus_by_name(row["corpus"])
            p = c.get_paper_by_id(row["paper_id"])

            idx = coord2idx[(c.name, p.id, row["sentence_id"])]
            text = row["sentence"]
            sec = row["section"]

            prev_sent_id = int(row["sentence_id"]) - 1
            next_sent_id = int(row["sentence_id"]) + 1

            # get previous sentence
            if prev_sent_id in p.content["id"].values:
                prev_doc = p.content.loc[prev_sent_id]
                prev_text = prev_doc["sentence"]
                prev_sec = prev_doc["section"]

            else:
                prev_text = ""
                prev_sec = ""

            # get next sentence
            if next_sent_id in p.content["id"].values:
                next_doc = p.content.loc[next_sent_id]
                next_text = next_doc["sentence"]
                next_sec = next_doc["section"]
            
            else:
                next_text = ""
                next_sec = ""

            data.append({
                "text": text,
                "doc_id": idx,
                "paper_title" : p.title,
                "year": p.year,
                "section": sec,
                "prev_text": prev_text,
                "prev_section": prev_sec,
                "next_text": next_text,
                "next_section": next_sec,
                "label": ""
            })

        df_doccano = pd.DataFrame(data)
        df_doccano["label"] = "" * df_doccano.shape[0]

        return df_doccano

    def draw_random_idx_from_corpus(self, corpus:str, n:int)->List[str]:
        # get all the possible indexes to choose from
        idx_list = list(self.candidates[self.candidates["corpus"] == corpus]["paper_id"].unique())

        # exclude those that have already been annotated
        with open(self.annotated_idx_path, "r") as f:
            annotated_idx = json.load(f)
        
        idx_list = list(set(idx_list).difference(set(annotated_idx)))

        # draw n random indexes
        return random.sample(idx_list, n)

