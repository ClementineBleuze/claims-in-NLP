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

    def same_family_name(self, a2):
        if len(self.norm_names) > 0 and len(a2.norm_names) > 0:
            # we consider equality if the first name is the same
            return self.norm_names[0] == a2.norm_names[0]
        
    def __eq__(self, a2):
        return set(self.norm_names) == set(a2.norm_names)