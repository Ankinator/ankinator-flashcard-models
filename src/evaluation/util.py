from typing import List, Tuple

import numpy as np


def extract_strings(inp_collection: List[Tuple[int, List[str]]]) -> List[str]:
    """
        Select the first string from each tuple in the input collection, as evaluation example.
    """
    return [t[1][0] for t in inp_collection]


def add_dimension_for_processing(inp_collection: List[str]) -> List[List[str]]:
    """
        Adds a dimension around the input sentences List[str] -> List[List[str]]
    """
    return np.expand_dims(np.array(inp_collection), axis=1).tolist()
