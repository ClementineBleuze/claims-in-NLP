import pandas as pd
from typing import List
import os
from sklearn.metrics import cohen_kappa_score
from collections import Counter
import simpledorff
from itertools import combinations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class AnnotationTask:
    """A class that represents an annotation task. It contains the title of the task, the path to the annotation guidelines, the set of labels used for the task, the annotators who will perform the task."""

    def __init__(self, title:str, guidelines_path:str = None, labels:List[str] = None, multi_labels:List[str] = None, annotators:List[str] = None):

        self.title = title # title of the task to perform
        self.guidelines_path = guidelines_path # path to the annotation guidelines
        self.labels = labels # set of labels used for the task
        self.multi_labels = multi_labels # labels + possible binary combinations of labels
        self.annotators = annotators # list of annotators to perform the task

        if labels is not None:
            self.multi_labels = self.labels + [f"{l1}#{l2}" for l1 in labels for l2 in labels if l1 < l2 ]


class AnnotationSet:

    def __init__(self, anno_dir:str, task: AnnotationTask):

        self.anno_dir = anno_dir # directory containing the different annotation files
        self.task = task # AnnotationTask object

        # check that the annotation directory contains all the required files
        assert set(os.listdir(anno_dir)).issuperset(set(["admin.csv"] + [f"{anno}.csv" for anno in self.task.annotators])), f"{anno_dir} does not contain the correct annotation files (should be [anno].csv for all annotators)."

        self.annotations = None # DataFrame containing the annotations
        self.ia_metrics = {} # DataFrame containing the inter-annotator agreement scores
        self.stats = None # DataFrame containing the statistics of the annotations


        self.load_annotations()
        #self.compute_agreement_metrics()
        self.make_stats()

    def compute_agreement_metrics(self):

        self.compute_cohen_kappa()
        self.compute_krippendorff_alpha()
        self.compute_ia_confusion_matrices()

    def load_annotations(self):

        frames = []
        
        ## Load the different annotation files
        # admin file
        try:
            frames.append((pd.read_csv(self.anno_dir + "admin.csv")[["id", "text", "current_sentence_section", "previous_sentence_section", "previous_sentence", "next_sentence_section", "next_sentence"]]))
        except:
            try:
                frames.append((pd.read_csv(self.anno_dir + "admin.csv")[["idx", "text", "sec", "prev_sec", "prev_sent", "next_sec", "next_sent"]]))
            except:
                df = pd.read_csv(self.anno_dir + "admin.csv")[["id", "text", "prev_text", "prev_section", "next_text", "next_section"]]
                df = df.rename(columns={"id":"idx"})
                frames.append(df)


        for anno in self.task.annotators:
            anno_file = anno + ".csv"
            df = pd.read_csv(self.anno_dir + anno_file)[["label", "Comments"]].astype(str)
            # if multiple labels include NC, map to NC
            df["label"] = df["label"].apply(lambda x: "NC" if "NC" in x.split("#") else x)
            df["label"] = df["label"].apply(lambda x: "NC" if x == "nan" else x)
            df["Comments"] = df["Comments"].apply(lambda x: "" if x == "nan" else x)

            df = df.rename(columns={"label": f"label_{anno}", "Comments": f"comments_{anno}"})
            frames.append(df)

        # merge all dataframes
        df = pd.concat(frames, axis=1)
        # for col in df.columns:
        #     if col.startswith("label") or col.startswith("comments"):
        #         df[col] = df[col].astype(str)
        #         df[col] = df[col].replace("nan", "NC")

        self.annotations = df
    
    def compute_cohen_kappa(self):
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
                    labels_anno1 = self.annotations.loc[idx, f"label_{anno1}"].values.astype(str).tolist()
                    labels_anno2 = self.annotations.loc[idx, f"label_{anno2}"].values.astype(str).tolist()

                    # compute the Cohen's Kappa score
                    agreement = cohen_kappa_score(labels_anno1, labels_anno2)
                    li.append(agreement)
                elif i == j:
                    li.append(1.0)
                else:
                    li.append(iaa_scores[j][i])
            iaa_scores.append(li)

        self.ia_metrics["cohen_kappa"] = pd.DataFrame(iaa_scores, columns=self.task.annotators, index=self.task.annotators)

    def compute_krippendorff_alpha(self, split_multi_labels = False):
        
        ## convert the annotations df into suitable format for krippendorff alpha computation
        # df with columns (unit id, annotator id, label)

        data = []
        for i, row in self.annotations.iterrows():
            #doc = str(row["doccano_art_id"]) + "-" + str(row["sentence_id"])
            doc = row["idx"]
            for anno in self.task.annotators:

                if split_multi_labels:
                    labels = row[f"label_{anno}"].split('#')
                    if labels[0] == "":
                        continue
                    for label in labels:
                        data.append([doc, anno, label])

                else:
                    label = row[f"label_{anno}"]
                    if label == "":
                        continue

                    data.append([doc, anno, label])

        df = pd.DataFrame(data, columns=["doc", "anno", "label"])

        ## compute krippendorff's alpha

        alpha = simpledorff.calculate_krippendorffs_alpha_for_df(df, experiment_col="doc", annotator_col="anno", class_col="label")
        self.ia_metrics["krippendorff_alpha"] = alpha

    def compute_coincidences(self, split_multi_labels = False):

        anno_pairs = {}

        for i, row in self.annotations.iterrows():
            labels_ = [row[f"label_{anno}"]for anno in self.task.annotators]
            comb_list = list(combinations(labels_, 2))

            ## Handle multi-labels
            pairs = []
            for comb in comb_list:
                if split_multi_labels:

                    labels = self.task.labels
                    split_l1 = comb[0].split("#")
                    split_l2 = comb[1].split("#")
                    
                    if len(split_l1) == 1  and len(split_l2) == 1:
                        pairs.append(AnnotationSet.order_tuple(comb))
                        continue

                    for l1 in split_l1:
                        for l2 in split_l2:
                            pairs.append(AnnotationSet.order_tuple((l1, l2)))

                else:
                    pairs.append(AnnotationSet.order_tuple(comb))
                
            ## Count the anno_pairs
            for pair in pairs:
                if pair not in anno_pairs:
                    anno_pairs[pair] = 0
                anno_pairs[pair] += 1

        # we do not count the empty annotations
        anno_pairs =  {k:anno_pairs[k] for k in anno_pairs if k[0] != "" and k[1] != ""}

        return anno_pairs

    def plot_coincidence_matrices(anno_pairs, labels):
        ## Raw counts matrix
        cm = np.zeros((len(labels), len(labels)), dtype = int)
        for l1 in labels:
            for l2 in labels:
                if (l1, l2) in anno_pairs:
                    cm[labels.index(l1), labels.index(l2)] = int(anno_pairs[(l1, l2)])
                elif (l2, l1) in anno_pairs:
                    cm[labels.index(l1), labels.index(l2)] = int(anno_pairs[(l2, l1)])
                else:
                    cm[labels.index(l1), labels.index(l2)] = 0
        
        # if some entire row is filled with 0, remove it and remove the corresponding labels
        to_remove = []
        for i in range(len(labels)):
            if sum(cm[i, :]) == 0:
                to_remove.append(i)
        cm = np.delete(cm, to_remove, axis=0)
        cm = np.delete(cm, to_remove, axis=1)
        labels = [l for i, l in enumerate(labels) if i not in to_remove]
        xlabels = labels + ['Total']
        ylabels = labels+ ['Total']

        # do not take the last column into account for the heatmap
        vmin = cm.min()
        vmax = cm.max()

        #add a column for the total number of annotations
        cm = np.concatenate((cm, cm.sum(axis=1)[:, np.newaxis]), axis=1)
        # same as a row
        cm = np.concatenate((cm, cm.sum(axis=0)[np.newaxis, :]), axis=0)

        ax = sns.heatmap(cm, annot=True, fmt = "d", cmap = "Blues", xticklabels=xlabels, yticklabels=ylabels, vmin=vmin, vmax = vmax)
        ax.xaxis.tick_top()

        if len(labels) > 6:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize = 8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize = 8)

        plt.title("Coincidence matrix\n")
        plt.show()

        # normalize by the total row
        cm = cm / cm[-1, :]
        # transpose
        cm = cm.T
        # drop the last row
        cm = cm[:-1, :]
        ylabels = ylabels[:len(ylabels)-1]


        ax2 = sns.heatmap(cm, annot=True, fmt = ".2f", cmap = "Blues", xticklabels=xlabels, yticklabels=ylabels, vmin=0, vmax = 1)
        ax2.xaxis.tick_top()

        if len(labels) > 6:
            ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize = 8)
            ax2.set_yticklabels(ylabels, fontsize = 8)
                
        plt.title("Coincidence ratios\n")
        plt.show()

        

    def order_tuple(t):
        return (t[0], t[1]) if t[0] < t[1] else (t[1], t[0])
    

    def compute_ia_confusion_matrices(self):
        cms = []

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
                        li.append(None)
                        continue

                    # get the labels for the two annotators
                    labels_anno1 = self.annotations.loc[idx, f"label_{anno1}"].values
                    labels_anno2 = self.annotations.loc[idx, f"label_{anno2}"].values


                    # compute the confusion matrix
                    cm = confusion_matrix(labels_anno1, labels_anno2, labels = self.task.labels)
                    li.append(cm)

                elif j < i:
                    cm = cms[j][i]
                    if cm is not None:
                        cm = cm.T
                    li.append(cm)

                elif i == j:
                    li.append(None)
                
            cms.append(li)
        

        self.ia_metrics["confusion_matrices"] = cms

    def plot_confusion_matrix_for_anno_pair(self, idx1, idx2):
        anno1, anno2 = self.task.annotators[idx1], self.task.annotators[idx2]

        cm = self.ia_metrics["confusion_matrices"][idx1][idx2]
        if cm is None:
            print(f"No confusion matrix available for {anno1} and {anno2}")
            return

        ax = sns.heatmap(cm, annot=True, fmt = "d", cmap = "Blues", xticklabels=self.task.labels, yticklabels=self.task.labels)
        ax.xaxis.tick_top()
        plt.xlabel(f"{anno2} labels")
        plt.ylabel(f"{anno1} labels")
        plt.title(f"Confusion matrix between {anno1} and {anno2}\n")
        plt.show()

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

    def get_ambiguous_annotations(self, anno_list = None):
        """A method that returns all ambiguous annotations, that is, those which have been commented by one or more user,
        or those where at least two annotators disagree on the correct label"""
        if anno_list is None:
            anno_list = self.task.annotators

        df = self.annotations

        # different annotations
        for i, row in df.iterrows():
            all_labels = []
            all_comments = []

            for anno in anno_list:

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

    def get_consensual_annotations(self):
        """Return the annotations where there is a strict majority to agree on a given label."""

        data = []
        for i, row in self.annotations.iterrows():
            labels = [row[f"label_{anno}"] for anno in self.task.annotators]
            # sort by counts
            count = Counter(labels)
            labels_count = [(k, v) for k, v in sorted(count.items(), key = lambda item: item[1], reverse = True)]

            if len(labels_count) == 1: # all annotators agree
                data.append([row["idx"], row["text"], labels[0]])

            elif labels_count[0][1] > labels_count[1][1]: # there is a strict majority
                data.append([row["idx"], row["text"], labels_count[0][0]])

        df = pd.DataFrame(data, columns = ["idx", "text", "label"])
        return df


