from typing import List, Tuple, Dict

from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize

from src.evaluation.evaluator import Evaluator
from src.evaluation.util import extract_strings, add_dimension_for_processing


class MeteorEvaluator(Evaluator):

    def __init__(self, save_to_file=True):
        self.sentences_from_reference: List[str] = []
        self.sentences_from_model: List[str] = []
        self.scores = None
        self.save_to_file = save_to_file

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:

        self.sentences_from_model = extract_strings(model_output)
        self.sentences_from_reference = extract_strings(references)

        sentences_from_model_tokenized =
        sentences_from_reference_tokenized =

