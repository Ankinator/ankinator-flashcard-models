import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from nltk.translate import meteor_score

from src.evaluation.metrics.evaluator import Evaluator
from src.evaluation.util import extract_strings, tokenize_list


class MeteorEvaluator(Evaluator):

    def __init__(self, save_to_file=True):
        super().__init__(save_to_file=save_to_file)
        self.scores: List[float] = []
        self.save_to_file = save_to_file

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:

        self.sentences_from_model = extract_strings(model_output)
        self.sentences_from_reference = extract_strings(references)

        sentences_from_model_tokenized = tokenize_list([l[0] for l in self.sentences_from_model])
        sentences_from_reference_tokenized = [tokenize_list(refs) for refs in self.sentences_from_reference]

        for i in range(len(sentences_from_model_tokenized)):
            self.scores.append(
                meteor_score.meteor_score(
                    references=sentences_from_reference_tokenized[i],
                    hypothesis=sentences_from_model_tokenized[i]
                )
            )

        if self.save_to_file:
            self.save_scores_to_file()

        np_scores = np.array(self.scores)

        return {
            "avg_sem_meteor": np_scores.mean(),
            "max_sem_meteor": np_scores.max(),
            "min_sem_meteor": np_scores.min()
        }

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model_out": self.sentences_from_model,
            "reference": self.sentences_from_reference,
            "max_sem_meteor": self.scores
        })

    def save_scores_to_file(self, path="out/eval/sem_meteor.csv"):

        if not os.path.exists(path):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        self.get_dataframe().to_csv(path_or_buf=path, index=False)
