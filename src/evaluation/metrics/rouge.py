from typing import Dict, List, Tuple

from torchmetrics.text.rouge import ROUGEScore
import pandas as pd
import os
import torch

from src.evaluation.metrics.evaluator import Evaluator
from src.evaluation.util import extract_strings


class RougeScoreEvaluator(Evaluator):

    def __init__(self, save_to_file=True):
        super().__init__(save_to_file=save_to_file)
        self.rouge_score = ROUGEScore()
        self.scores: Dict[str, torch.Tensor] = {}

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:

        self.sentences_from_model = extract_strings(inp_collection=model_output)
        self.sentences_from_reference = extract_strings(inp_collection=references)

        self.scores = self.rouge_score(self.sentences_from_model, self.sentences_from_reference)
        return {key: t.item() for key, t in self.scores.items()}

    def save_scores_to_file(self, path):


    def get_dataframe(self) -> pd.DataFrame:
        pass


