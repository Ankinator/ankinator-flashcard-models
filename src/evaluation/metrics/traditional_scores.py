import os
import string
from typing import List, Tuple, Dict, Set

import nltk
import numpy as np
import pandas as pd
from nltk.metrics.scores import precision, recall, f_measure
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.evaluation.util import extract_strings
from src.evaluation.metrics.evaluator import Evaluator


class PRFScoreEvaluator(Evaluator):

    def __init__(self, save_to_file=True):
        super().__init__(save_to_file)
        self.precision_scores: List[float] = []
        self.recall_scores: List[float] = []
        self.f_measure_scores: List[float] = []

        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
        self.stemmer = PorterStemmer()

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:
        self.sentences_from_model = extract_strings(model_output)
        self.sentences_from_reference = extract_strings(references)

        normalized_from_model = self._normalize_and_convert_to_set(self.sentences_from_model)
        normalized_from_reference = self._normalize_and_convert_to_set(self.sentences_from_reference)

        self.precision_scores = [precision(reference=r, test=m) if precision(reference=r, test=m) is not None else 0 for r, m in
                                 zip(normalized_from_reference, normalized_from_model)]
        self.recall_scores = [recall(reference=r, test=m) for r, m in
                              zip(normalized_from_reference, normalized_from_model)]
        self.f_measure_scores = [f_measure(reference=r, test=m) if f_measure(reference=r, test=m) is not None else 0 for r, m in
                                 zip(normalized_from_reference, normalized_from_model)]

        if self.save_to_file:
            self.save_scores_to_file()

        p_scores = np.array(self.precision_scores)
        r_scores = np.array(self.recall_scores)
        f_scores = np.array(self.f_measure_scores)

        return {
            "avg_p": p_scores.mean(),
            "max_p": p_scores.max(),
            "min_p": p_scores.min(),
            "avg_r": r_scores.mean(),
            "max_r": r_scores.max(),
            "min_r": r_scores.min(),
            "avg_f1": f_scores.mean(),
            "max_f1": f_scores.max(),
            "min_f1": f_scores.min()
        }

    def _normalize_and_convert_to_set(self, inp_collection: List[str]) -> List[Set[str]]:
        """
            Converts a list containing a list of tokens to a list including sets normalized of tokens
        """
        tokenized = [word_tokenize(text.lower()) for text in inp_collection]
        tokenized_without_stopwords = [[t for t in tokens if t not in self.stopwords] for tokens in tokenized]
        without_punctuations = [[t for t in tokens if t not in string.punctuation] for tokens in
                                tokenized_without_stopwords]
        stemmed = [[self.stemmer.stem(t) for t in tokens] for tokens in without_punctuations]
        return [set(tokens) for tokens in stemmed]

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model_out": self.sentences_from_model,
            "reference": self.sentences_from_reference,
            "p": self.precision_scores,
            "r": self.recall_scores,
            "f1": self.f_measure_scores
        })

    def save_scores_to_file(self, path="out/eval/traditional_scores.csv"):
        if not os.path.exists(path):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        self.get_dataframe().to_csv(path_or_buf=path, index=False)
