from typing import List, Tuple

import numpy as np
from nltk import word_tokenize


def extract_strings(inp_collection: List[Tuple[int, List[str]]]) -> List[List[str]]:
    """
        Selects the strings from each tuple in the input collection, as evaluation example.
    """
    return [t[1] for t in inp_collection]


def tokenize_list(list: List[str]) -> List[List[str]]:
    """
        Tokenizes a list of strings using nltks tokenize function
    """
    return [word_tokenize(txt) for txt in list]


def add_dimension_for_processing(inp_collection: List[str]) -> List[List[str]]:
    """
        Adds a dimension around the input sentences List[str] -> List[List[str]]
    """
    return np.expand_dims(np.array(inp_collection), axis=1).tolist()
