from typing import List
import re

class Author:
    """A class to represent an author of a research paper"""

    SCHAR_MAPPING = {
    'ã': 'a',
    'ú': 'u',
    'í': 'i',
    'ğ': 'g',
    'á': 'a',
    'ć': 'c',
    'Ž': 'Z',
    'İ': 'I',
    'î': 'i',
    'ü': 'u',
    'š': 's',
    'ý': 'y',
    'ò': 'o',
    'Š': 's',
    'é': 'e',
    'Ø': 'O',
    'ń': 'n',
    'à': 'a',
    'ă': 'a',
    'æ': 'ae',
    'Á': 'A',
    'ñ': 'n',
    'ç': 'c',
    'ř': 'r',
    'Č': 'C',
    'Ş': 'S',
    'è': 'e',
    'ı': 'i',
    'ä': 'a',
    'ě': 'e',
    'ł': 'l',
    'ô': 'o',
    'Ç': 'C',
    'ó': 'o',
    'ö': 'o'
    }

    def __init__(self, names:List[str], others:str = None):
        self.names = names # list of author names (str)
        self.norm_names = [] # list of author names after normalization of special characters (str)
        self.others = others # other information about the author (str) e.g affiliation, email, etc.

    def to_dict(self):
        return {
            "names": self.names,
            "norm_names": self.norm_names,
            "others": self.others
        }
    
    @classmethod
    def from_dict(cls, d:dict):
        return cls(names = d["names"], others = d["others"])

    def normalize_names(self):
        """Normalize the author names by mapping special characters to their corresponding "basuc" characters"""
        norm_names = []

        for name in self.names:
            # catch things like {\textcommabelow{S}} and replace with S
            pat1 = re.compile(r"(\{\\textcommabelow\{(.)\}{2})")
            name = re.sub(pat1, r"\2", name)

            # catch encodings of accents, e.g {\\'\\i} or {\\v{c}} or {\'e} and replace with the simple letter
            pat2 = re.compile(r"(\{\\?.[\\\{]?(.)\}{1,2})")
            name = re.sub(pat2, r"\2", name)

            # catch apostrophes
            pat3 = re.compile(re.compile(r"\{'\}"))
            name = re.sub(pat3, "'", name)

            # catch special characters
            for char in name:
                if char in Author.SCHAR_MAPPING:
                    name = name.replace(char, Author.SCHAR_MAPPING[char])
            
            norm_names.append(name)
        
        self.norm_names = norm_names

    def normalize_str(s:str):
        """Normalize a string by mapping special characters to their corresponding "basic" characters"""
        
        pat1 = re.compile(r"(\{\\textcommabelow\{(.)\}{2})")
        s = re.sub(pat1, r"\2", s)

        pat2 = re.compile(r"(\{\\?.[\\\{]?(.)\}{1,2})")
        s = re.sub(pat2, r"\2", s)

        pat3 = re.compile(re.compile(r"\{'\}"))
        s = re.sub(pat3, "'", s)

        for char in s:
            if char in Author.SCHAR_MAPPING:
                s = s.replace(char, Author.SCHAR_MAPPING[char])
        
        return s

    def same_family_name(self, a2):
        if len(self.norm_names) > 0 and len(a2.norm_names) > 0:
            # we consider equality if the first name is the same
            return self.norm_names[0] == a2.norm_names[0]
        
    def __eq__(self, a2):
        return set(self.norm_names) == set(a2.norm_names)