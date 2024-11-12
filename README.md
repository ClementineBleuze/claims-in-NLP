# Analysing claims in NLP papers
This repository contains work conducted during a 6-month internship at LORIA (supervisors: Karën FORT, Maxime AMBLARD, Fanny DUCEL) about the analysis of claims in NLP research, between March-August 2024. It is complementary with an additional [HuggingFace repository](https://huggingface.co/datasets/ClementineBleuze/CNP) containing files that are too large to fit here, and a [HuggingFace checkpoint](https://huggingface.co/ClementineBleuze/scibert_prefix_cont_ll_SEP) from which you can download our best model for claim category classification (Weighted F1-score = 0.90). You can also read the related [Master's thesis](M2_thesis_BLEUZE_Clementine.pdf).

## Structure of the repository  
The repository is organised in multiple folders corresponding to the consecutive steps of our work : 
- First, we gathered papers originating from the ACL Anthology (we reused the [ACL OCL corpus](https://github.com/shauryr/ACL-anthology-corpus) and from ArXiv. We parsed their content and stored sentences from the full text: see Folder [1-corpus-constitution](1-corpus-constitution).
- Second, we conducted multiple annotation rounds on the corpus of sentences to validate different versions of a taxonomy of claims: see Folder [2-claims-annotation](2-claims-annotation).
- Third, we trained models to predict the correct claim category of a given sentence. Our best model is available on [Huggingface](https://huggingface.co/ClementineBleuze/scibert_prefix_cont_ll_SEP). We ran this model for inference on the entire corpus and also used off-the-shelf models of [Pei and Jurgens](https://github.com/Jiaxin-Pei/Certainty-in-Science-Communication) to predict sentence-level and aspect-level certainty for all the sentences: see Folder [3-claims-classification](3-claims-classification).
- Finally, we conducted multiple analyses on the collected annotations: see Folder [4-corpus-analysis](4-corpus-analysis).
  
## How to access data files ?  
Data files are available on [Huggingface](https://huggingface.co/datasets/ClementineBleuze/CNP/tree/main/data). It includes: 
- `papers.csv`: all the metadata about the papers included in our corpus
- `sentences.csv`: all the sentences of the corpus + their annotations
- XML files of the papers of the corpus (obtained after parsing their PDF versions) 

## Contact  
If you have requests or questions about this repository, please contact **clementine.bleuze@univ-lorraine.fr**. 
