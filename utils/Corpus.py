from collections import Counter
import pandas as pd
import tqdm
import utils.Paper
import spacy

class Corpus:

    def __init__(self, xml_dir_path, metadata_path, name=None):

        self.name = name # name of the corpus (str: ACL or arXiv)
        self.xml_dir_path = xml_dir_path # path to the directory containing the XML files of the corpus papers(str)
        self.metadata_path = metadata_path # path to the metadata file of the corpus (str)
        self.model = Corpus.init_spacy_model() # spacy model for NLP processing

        self.papers = [] # list of Paper objects (will be populated using the load_papers method)
        self.papers_with_errors = [] # list of Paper objects that could not be loaded

    def init_spacy_model():
        """Initialize the spacy model for NLP processing"""
        nlp = spacy.load("en_core_web_sm")

        # remove the default prefix and suffix regexes to allow for the use of square brackets in the text
        # avoid wrong sentence segmentation because of reference format
        prefixes = list(nlp.Defaults.prefixes)
        prefixes.remove("\[")
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search

        suffixes = list(nlp.Defaults.suffixes)
        suffixes.remove("\]")
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        nlp.tokenizer.suffix_search = suffix_regex.search

        return nlp
    
    def get_paper_by_id(self, id:str):
        """Get a paper by its id"""
        # we assume that the id is unique
        return [p for p in self.papers if p.id == id][0]

    def load_papers(self):
        """A method to create Paper objects from metadata information and corresponding xml files,
        and store them in the papers attribute of the corpus object"""

        # Load the metadata file
        if self.metadata_path.endswith(".csv"):
            df = pd.read_csv(self.metadata_path, encoding = "utf-8")
        elif self.metadata_path.endswith(".pkl"):
            df = pd.read_pickle(self.metadata_path)

        # Load the papers
        for i, row in tqdm.tqdm(df.iterrows(), total = len(df)):
            paper = utils.Paper.Paper(d = row.to_dict(), c = self)
            if paper.init_error is None:
                self.papers.append(paper)
            else:
                self.papers_with_errors.append(paper)

    def initialize(self, verbose = False):
        """Initialize the corpus by loading the papers and preprocessing their content"""

        if verbose:
            print(f"Corpus '{self.name}' is being initialized...\n")

        self.load_papers()

        if verbose:
            print(f"Corpus '{self.name}' papers have been loaded.")
            self.describe()
            print("\nPreprocessing papers...")
        
        self.preprocess_papers()

    def preprocess_papers(self):
        """Preprocess the papers of the corpus by classifying the sections of each paper into candidates vs. non-candidates for containing claims"""

        for paper in self.papers:
            paper.preprocess_content_sections()

    def describe(self, error_verbose = False):
        """Describe the corpus in terms of number of papers and errors encountered during the loading process"""
        nb_ok, nb_errors = len(self.papers), len(self.papers_with_errors)

        print(f"Corpus '{self.name}' was filled with {nb_ok + nb_errors} papers:")
        print(f"  - {nb_ok} papers were successfully loaded")
        print(f"  - {nb_errors} papers could not be loaded")

        if error_verbose:
            print("\nErrors:")
            errors = [p.init_error for p in self.papers_with_errors]
            for counter_row in Counter(errors).items():
                print(f"  - {counter_row[0]} : {counter_row[1]}")


    