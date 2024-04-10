import utils.Corpus
import utils.Author
import pandas as pd
import tqdm
import json
import xml.etree.ElementTree as ET
import re
from typing import List, Tuple
from nltk.tokenize import sent_tokenize

class Paper:
    """A class to represent a research paper"""

    # keywords (or subwords) that are indicative of a section containing claims
    CAND_SECTIONS = ["introduction", "motivation", "contribution", "result", "finding", "analysis", "observation"
                 "evaluation", "perform", "conclusion", "discussion", "limit", "ethic", "challenge", "future", 
                 "inter-annotator", "impact", "directions", "comparison"]
    
    # keywords (or subwords) that are indicative of a section not containing claims
    NON_CAND_SECTIONS = ["related", "prior", "study", "studies", "background", "overview", "state-of-the-art",
                     "algorithm", "method", "model", "setup", "setting", "experiment", "parameter", "hyperparameter",
                     "task", "training", "architecture", "implement", "corpus", "corpora", "data", "description",
                     "definition", "feature", "example", "acknowledgement", "reference", "appendix", "supplement", 
                     "problem", "process", "framework", "current", "tuning"]

    def __init__(self, d: dict = None, c = None):
        """Initialize the paper from a metadata dictionary and a corpus object"""

        # Paper information
        self.title = None
        self.id = None
        self.authors = None
        self.year = None
        self.publisher = None
        self.category = None
        self.num_citedby = None
        self.corpus = None
        self.xml_path = None
        self.corpus = None

        # Paper content
        self.abstract = None
        self.content = None
        self.sections = None
        self.nb_ref = None

        # Errors
        self.init_error = None
        
        if c is not None:
            self.corpus = c

            ## ACL corpus
            if self.corpus.name == "ACL":

                if d is not None:
                    try:
                        self.title = Paper.clean_text(d["title"])
                        self.id = d["acl_id"]
                        self.authors = self.extract_authors_from_str(d["author"])
                        self.year = d["year"]
                        self.publisher = d["publisher"]
                        self.category = d["Model Predicted Topics"]
                        self.num_citedby = d["numcitedby"]
                        self.abstract = d["abstract"]
                    except:
                        print(f"Error: check that the metadata dictionary used to build the Paper Object contains the following columns: title, acl_id, author, year, publisher, Model Predicted Topics, numcitedby, abstract")

                    #XML files are named {xml_dir_path}/{id}.tei.xml
                    self.xml_path = f"{c.xml_dir_path}/{self.id}.tei.xml"

            # ArXiv corpus
            elif self.corpus.name == "arXiv":

                if d is not None:
                    try:
                        self.title = d["title"]
                        self.id = d["id"]
                        self.authors = self.extract_authors_from_str(d["authors_parsed"])
                        self.publisher = None
                        self.abstract = d["abstract"]

                        ## LATER
                        # self.year = 
                        # self.category =
                        # self.num_citedby =
                    
                    except:
                        print(f"Error: check that the metadata dictionary used to build the Paper Object contains the following columns: title, id, authors_parsed, publisher, abstract")
                    
                    # XML files can be named in different ways, depending on their id structure:
                    # first, we need to know the last version available for each paper
                    versions = json.loads(d["versions"].replace("\'", "\""))
                    last_version = versions[-1]["version"]

                    if "." in self.id:
                        self.xml_path = f"{c.xml_dir_path}/{self.id}{last_version}.grobid.tei.xml"
                    elif "/" in self.id:
                        category, id = self.id.split("/")[:2]
                        self.xml_path = f"{c.xml_dir_path}/{id}{last_version}.grobid.tei.xml"

            # Other corpus       
            else:
                pass
                
            self.content, self.sections, self.nb_ref, self.init_error = self.load_content_from_xml(self.xml_path, self.abstract)


    def load_content_from_xml(self, xml_file:str, abstract:str) -> Tuple[pd.DataFrame, dict, int, str]:
        """Load the content of the paper from an XML file, and return the content as a DataFrame, the sections as a dictionary, the number of references, and an error message if any"""
        
        # init
        content, sections, nb_ref, error = None, None, None, None

        ## Read the XML file
        try:
            with open(xml_file, encoding = "utf-8") as f:
                content = f.read()
        except FileNotFoundError as e:
            error = f"FileNotFoundError: XML file does not exist"
            return content, sections, nb_ref, error

        # we exclude papers that have bad input data
        if content.startswith("[BAD_INPUT_DATA]"):
            error = "Parsing error: XML file not well formed"
            return content, sections, nb_ref, error

        ## Parse the XML file

        # Remove all ref tags but keep their content (<ref>content</ref> -> content)
        ref_pattern = re.compile(r"<ref.*?>(.*?)</ref>")
        content = re.sub(ref_pattern, r"\1", content)

        try: 
            root = ET.fromstring(content)
        except:
            error = "Parsing error: XML file not well formed"
            return content, sections, nb_ref, error
        
        # Language check: we exlude the papers that are not in english
        for at in list(root[1].attrib.keys()):
            if "lang" in str(at):
                if root[1].attrib[at] != "en":
                    error = f"Noisy data: wrong language ({root[1].attrib[at]})"
                    return content, sections, nb_ref, error

        ## Parsing the XML file if there is no error
        sections = {}
        content = []
        nb_sections = 0
        nb_sentences = 0
        abs_sentences = []

        # start with the abstract
        found_abstract = False

        if abstract is not None: # case where the abstract is already known from the metadata
            abs_sentences = sent_tokenize(abstract)
            found_abstract = len(abs_sentences) > 0

        if not found_abstract: # cases when the abstract was not provided in the metadata file, but we try to find it in the XML file
            for c in root[0]:
                if "profileDesc" in c.tag:
                    for d in c:
                        if "abstract" in d.tag:
                            if len(d) > 0: #d[0] is the abstract div
                                if d[0].text: # abstract div
                                    abs_sentences = sent_tokenize(d[0].text)
                                    found_abstract = len(abs_sentences) > 0
                                    break

                                elif len(d[0]) > 0: # get all paragraphs
                                    text = "\n".join([c.text for c in d[0][1:]])
                                    abs_sentences = sent_tokenize(text)
                                    found_abstract = len(abs_sentences) > 0
                                    break
            
        if found_abstract:
            for i, sentence in enumerate(abs_sentences):
                content.append({"id": i, "sentence": Paper.clean_text(sentence), "section": "abstract"})
            nb_sentences += len(abs_sentences)
        else:
            error = "Parsing error: no abstract found"
            return content, sections, nb_ref, error

        ## Parse paper content
        for child in root[1][0]: # root.text.body
            # check the <div> (sections identified by grobid) because they indicate the sections of the paper (but also figures or notes)
            if "div" in child.tag:
                if len(child) > 0: # if not empty
                    header, n, head_n = child[0].text, None, None

                    # we do not want to keep the figures and tables
                    if not header.lower().startswith("figure") and not header.lower().startswith("table"):

                        # extract the textual content of the section
                        if len(child) > 0 :
                            text = "\n".join([Paper.clean_text(c.text) for c in child[1:]])

                            # in case the section content is not acceptable, we skip it
                            if not Paper.is_acceptable_section_content(text):
                                continue
                            
                            # in case the section header is too long, we consider it as part of the content
                            if not Paper.is_acceptable_section_header(header):
                                if Paper.get_alpha_numerical_ratio(header) > 0.5:
                                    text = header + "\n" + text 
                                header = "unidentified-section"

                            # split the content into sentences
                            sentences = sent_tokenize(text)

                            for i, sentence in enumerate(sentences):
                                if Paper.is_acceptable_sentence(sentence):
                                    content.append({"id": nb_sentences, "sentence": Paper.clean_text(sentence), "section": header})
                                    nb_sentences += 1
                            
                            # in case no sentence was finally added, we skip the section
                            if nb_sentences == len(abs_sentences):
                                continue


                        # check if the section if numbered
                        if "n" in list(child[0].attrib):
                            n = child[0].attrib["n"]
                            # check if this section is actually a subsection
                            head_n = re.search(re.compile("(.*)\.\d"), n)
                            if head_n:
                                head_n = head_n.group(1)
                        
                        # update the sections oranisation
                        sections[nb_sections] = {"n": n, "header": header, "head_n": head_n}
                        nb_sections += 1
                                         

        content = pd.DataFrame(content)

        # check if empty
        if content.empty:
            return content, sections, nb_ref, "parsing error: no paper content found"

        ## Count the number of references
        for child in root[1][1]:
            if ("type" in child.attrib) and (child.attrib["type"] == "references"):
                nb_ref = len(child[0])
                break

        if content.section.nunique() < 2:
            error = "parsing error: not enough paper content found (<2 distinct sections)"

        return content, sections, nb_ref, error
    
    def extract_authors_from_str(self, s:str)->List[str]:
        """Extract the authors from a string"""
        authors = []

        if s:
            if self.corpus.name == "ACL":
                s = s.replace("\n", " ")

                authors_s = s.split(" and ")
                for a_s in authors_s:
                    
                    names = a_s.split(",")
                    others = None # additionnal info

                    # if len(names) > 2:
                    #     others = names[2:]
                    #     names = names[:2]

                    # create the author object
                    a = utils.Author.Author(names = [n.strip() for n in names], others = others)

                    # normalize the author names
                    a.normalize_names()

                    authors.append(a)

            elif self.corpus.name == "arXiv":

                authors_s = s.split("],")

                for a_s in authors_s:
                    a_s = a_s.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")

                    names = a_s.split(",")
                    others = None # additionnal info

                    if len(names) > 2:
                        others = names[2:]
                        names = names[:2]
                    
                    # create the author object
                    a = utils.Author.Author(names = [n.strip() for n in names], others = others)

                    # normalize the author names
                    a.normalize_names()

                    authors.append(a)
                    
            else:
                return authors


        return authors

    def preprocess_content_sections(self):
        """A method that classifies the sections of the paper into candidates vs. non-candidates for containing claims,
        based on the presence of certain keywords in the section headers (and their root sections)"""
        df = self.content.copy()

        df["candidate"] = True # by default, all sections are considered as candidates
        section_headers = df.section.unique()

        for i, sh in enumerate(section_headers):
            si = i - 1 # section index in the sections dictionary (-1 because of the abstract)
            # if the header contains a keyword that is indicative of a candidate section or is the abstract, we keep it as candidate
            if sh == "abstract" or Paper.are_words_in_string(Paper.CAND_SECTIONS, sh.lower()):
                continue   
            
            # if the header contains a keyword that is indicative of a non-candidate section, we mark it as non-candidate
            elif Paper.are_words_in_string(Paper.NON_CAND_SECTIONS, sh.lower()):
                df.loc[df.section == sh, "candidate"] = False
            
            # if the header is not informative, we check the root section
            else: 
                if self.sections[si]["head_n"]:
                    # if the structure of the sections is coherent, the root must already have been processed
                    hi = self.get_root_section_idx(si)
                    if hi:
                        hh = self.sections[hi]["header"]
                        # if the root section is a non-candidate, we mark the section as non-candidate
                        df.loc[df.section == sh, "candidate"] = df.loc[df.section == hh, "candidate"].values[0]        

        self.content = df

    def get_similar_candidates(self):
        pass
    

    def get_subsections_idx(self, section_idx:int):
        """Get the ids of the subsections of a section whose id is provided"""
        subsections_idx = []
        n = self.sections[section_idx]["n"]

        for i in self.sections:
            head_n = self.sections[i]["head_n"]
            if head_n:
                if head_n.startswith(n):
                    subsections_idx.append(i)

        return subsections_idx
    
    def get_root_section_idx(self, section_idx:int):
        """Get the id of the root section of a section whose id is provided"""
        head_n = self.sections[section_idx]["head_n"]
        if head_n:
            for i in self.sections:
                if self.sections[i]["n"] == head_n or self.sections[i]["n"] == head_n + ".":
                    return i
        return None


    def are_words_in_string(li: List[str], s: str):
        for word in li:
            if word in s:
                return True
        return False
    
    def clean_text(text:str, to_ascii = False)->str:
        # remove \n and \t
        text = text.replace("\n", " ").replace("\t", " ")

        # keep only ascii characters ?
        if to_ascii:
            text = text.encode("ascii", "ignore").decode("utf-8")

        # if html tags are present, just keep the text inside
        text = re.sub(re.compile(r"<.*?>(.*)</.*?>"), "", text)

        return text

    def get_alpha_numerical_ratio(text:str)->float:

        # remove excess spaces
        text = re.sub(r"\s+", " ", text)
        alpha_num = 0
        for char in text:
            if char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                alpha_num += 1
        return alpha_num / len(text) if len(text) > 0 else 0
    
    def is_acceptable_section_header(header: str):
        """Reject as non acceptable the headers that are too long or have a low alpha numerical ratio"""
        return Paper.get_alpha_numerical_ratio(header) > 0.5 and len(header) < 100
    
    def is_acceptable_section_content(content: str):
        """Reject as non acceptable the content that has a low alpha numerical ratio and is too short"""
        return Paper.get_alpha_numerical_ratio(content) > 0.5 and len(content) > 100
    
    def is_acceptable_sentence(sentence: str):
        """Reject as non acceptable the sentences that have a low alpha numerical ratio and are too short"""
        return Paper.get_alpha_numerical_ratio(sentence) > 0.5 and len(sentence) > 10


            