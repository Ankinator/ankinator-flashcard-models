import unittest

from tests.evaluation import build_synthetic_model_outputs
from src.evaluation.metrics.traditional_scores import PRFScoreEvaluator


class TestTraditionalScores(unittest.TestCase):

    def test_p_r_f1_calculation(self):
        model_outs, references = build_synthetic_model_outputs()
        prf_score_evaluator = PRFScoreEvaluator(save_to_file=False)
        self.assertDictEqual(
            {'avg_p': 0.55, 'max_p': 1.0, 'min_p': 0.25, 'avg_r': 0.42333333333333334, 'max_r': 0.6666666666666666,
             'min_r': 0.2, 'avg_f1': 0.4749206349206349, 'max_f1': 0.8, 'min_f1': 0.2222222222222222},
            prf_score_evaluator(model_output=model_outs, references=references)
        )


