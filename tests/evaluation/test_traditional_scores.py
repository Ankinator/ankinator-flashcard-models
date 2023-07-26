import os
import unittest

import pandas as pd

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

    def test_write_to_file(self):
        path = "out/eval/traditional_scores.csv"
        model_outs, references = build_synthetic_model_outputs()

        prf_score_evaluator = PRFScoreEvaluator(save_to_file=True)
        prf_score_evaluator(model_output=model_outs, references=references)

        self.assertTrue(os.path.exists(path))
        file_contents_df = pd.read_csv(path)
        self.assertListEqual(list1=file_contents_df.columns.to_list(),
                             list2=["model_out", "reference", "max_p", "max_r", "max_f1"])

