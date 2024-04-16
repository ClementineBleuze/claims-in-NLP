import pandas as pd
from typing import List
import os
from sklearn.metrics import cohen_kappa_score
from collections import Counter


class AnnotationTask:

    def __init__(self, title:str, guidelines_path:str = None, labels:List[str] = None, annotators:List[str] = None):

        self.title = title # title of the task to perform
        self.guidelines_path = guidelines_path # path to the annotation guidelines
        self.labels = labels # set of labels used for the task
        self.annotators = annotators # list of annotators to perform the task


class AnnotationSet:

    def __init__(self, anno_dir:str, task: AnnotationTask):

        self.anno_dir = anno_dir # directory containing the different annotation files
        self.task = task # AnnotationTask object

        assert set(os.listdir(anno_dir)).issuperset(set(["admin.csv"] + [f"{anno}.csv" for anno in self.task.annotators])), f"{anno_dir} does not contain the correct annotation files (should be [anno].csv for all annotators)."

        self.annotations = None # DataFrame containing the annotations
        self.iaa_scores = None # DataFrame containing the inter-annotator agreement scores
        self.stats = None # DataFrame containing the statistics of the annotations

        self.load_annotations()
        self.compute_iaa()
        self.make_stats()

    def load_annotations(self):

        frames = []

        # admin
        frames.append((pd.read_csv(self.anno_dir + "admin.csv")[["doccano_art_id", "sentence_id", "text", "current_sentence_section", "previous_sentence_section", "previous_sentence", "next_sentence_section", "next_sentence"]]))

        for anno in self.task.annotators:
            anno_file = anno + ".csv"
            df = pd.read_csv(self.anno_dir + anno_file)[["label", "Comments"]].astype(str)
            ## replace nan values by "NC" (No Claim)
            #df["label"] = df["label"].apply(lambda x: "NC" if x == "nan" else x)
            df = df.rename(columns={"label": f"label_{anno}", "Comments": f"comments_{anno}"})
            frames.append(df)

        # merge all dataframes
        df = pd.concat(frames, axis=1)
        for col in df.columns:
            if col.startswith("label") or col.startswith("comments"):
                df[col] = df[col].astype(str)
                df[col] = df[col].replace("nan", "")
        
        self.annotations = df
    
    def compute_iaa(self):
        """Compute the inter-annotator agreement scores using Cohen's Kappa coefficient."""

        iaa_scores = []

        for i, anno1 in enumerate(self.task.annotators):
            li = []
            for j, anno2 in enumerate(self.task.annotators):
                if i < j:
                    
                    # get the indices where the two annotators have annotated
                    anno1_idx = [i for i in range(self.annotations.shape[0]) if str(self.annotations.at[i, f"label_{anno1}"] )!= ""]
                    anno2_idx = [i for i in range(self.annotations.shape[0]) if str(self.annotations.at[i, f"label_{anno2}"]) != ""]
                    
                    # only consider the indices where both annotators have annotated
                    idx = list(set(anno1_idx) & set(anno2_idx))

                    if len(idx) == 0:
                        li.append(0.0)
                        continue

                    # get the labels for the two annotators
                    labels_anno1 = self.annotations.loc[idx, f"label_{anno1}"].values
                    labels_anno2 = self.annotations.loc[idx, f"label_{anno2}"].values

                    # compute the Cohen's Kappa score
                    agreement = cohen_kappa_score(labels_anno1, labels_anno2)
                    li.append(agreement)
                elif i == j:
                    li.append(1.0)
                else:
                    li.append(iaa_scores[j][i])
            iaa_scores.append(li)

        self.iaa_scores = pd.DataFrame(iaa_scores, columns=self.task.annotators, index=self.task.annotators)

    def make_stats(self):
        """Compute the statistics of the annotations."""

        stats = []
        for anno in self.task.annotators:
            # get all labels for this annotator
            all_labels = self.annotations[f"label_{anno}"].astype(str).values

            # frequency of each label
            li = []
            for label in self.task.labels:
                nb = len([l for l in all_labels if label in l.split('#')])
                li.append(nb)
            # total number of annotations
            total_nb = len([l for l in all_labels if l != ""])

            li.append(total_nb)

            # percentage of each label
            for i in range(len(li)-1):
                li.append(li[i]/total_nb) if total_nb != 0 else li.append(0)

            # percentage of annotations
            li.append(total_nb/len(all_labels))
            stats.append(li)

        columns =  self.task.labels + ['total'] + [f"{l}_rr" for l in self.task.labels] + ['completion_r']
        self.stats = pd.DataFrame(stats, columns=columns, index=self.task.annotators)

    def get_ambiguous_annotations(self):
        """A method that returns all ambiguous annotations, that is, those which have been commented by one or more user,
        or those where at least two annotators disagree on the correct label"""
        df = self.annotations

        # different annotations
        for i, row in df.iterrows():
            all_labels = []
            all_comments = []

            for anno in self.task.annotators:

                labels= df.at[i, f"label_{anno}"]
                if labels != "":
                    for label in labels.split("#"):
                        all_labels.append(label)

                comment = df.at[i, f"comments_{anno}"]
                if comment != "":
                    all_comments.append(comment)

            if len(list(set(all_labels))) >= 1: #there is no complete agreement
                counter = Counter(all_labels)
                maj_label = counter.most_common(1)[0][0]
                df.at[i, "maj_label"] = maj_label
            
            if len(list(set(all_labels))) == 1:
                df.at[i, "agreement"] = True
            else:
                df.at[i, "agreement"] = False

            if len(all_comments) != 0:
                df.at[i, "commented"] = True
            else:
                df.at[i, "commented"] = False


        return df[(df["agreement"] == False) | (df["commented"] == True)]



