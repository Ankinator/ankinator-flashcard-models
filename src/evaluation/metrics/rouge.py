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

        self.sentences_from_model = [i[0] for i in extract_strings(inp_collection=model_output)]
        self.sentences_from_reference = extract_strings(inp_collection=references)

        self.scores = self.rouge_score(self.sentences_from_model, self.sentences_from_reference)

        if self.save_to_file:
            self.save_scores_to_file()

        return {key: t.item() for key, t in self.scores.items()}

    def save_scores_to_file(self, path="out/eval/rouge_scores.csv"):
        if not os.path.exists(path):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        self.get_dataframe().to_csv(path_or_buf=path, index=False)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "rouge_keys": self.scores.keys(),
            "scores": [v.item() for v in self.scores.values()]
        })
