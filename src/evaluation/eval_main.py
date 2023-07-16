from typing import List, Tuple

from src.evaluation.metrics.grammatic_errors import LanguageToolEvaluator
from src.evaluation.metrics.rouge import RougeScoreEvaluator
from src.evaluation.metrics.traditional_scores import PRFScoreEvaluator
from src.evaluation.metrics.semantic_similarity import SentenceTransformerEvaluator
from src.evaluation.metrics.meteor_score import MeteorEvaluator


class Metrics:

    def __init__(self, save_to_file=True):
        self.lt_eval = LanguageToolEvaluator(save_to_file=save_to_file)
        self.rouge = RougeScoreEvaluator(save_to_file=save_to_file)
        self.prf = PRFScoreEvaluator(save_to_file=save_to_file)
        self.sbert = SentenceTransformerEvaluator(save_to_file=save_to_file)
        self.meteor = MeteorEvaluator(save_to_file=save_to_file)

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]):
        summary = {
            **self.lt_eval(model_output=model_output, references=references),
            **self.rouge(model_output=model_output, references=references),
            **self.prf(model_output=model_output, references=references),
            **self.sbert(model_output=model_output, references=references),
            **self.meteor(model_output=model_output, references=references)
        }
        return summary
