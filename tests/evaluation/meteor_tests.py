import os
import unittest

import pandas as pd

from tests.evaluation import build_synthetic_model_outputs
from src.evaluation.meteor_score import MeteorEvaluator


class TestMeteor(unittest.TestCase):

    def test_meteor_calculation(self):
        model_outs, references = build_synthetic_model_outputs()
        meteor_eval = MeteorEvaluator(save_to_file=False)
        self.assertDictEqual(
            {'avg_sem_meteor': 0.35021226102225567, 'max_sem_meteor': 0.7577355836849509,
             'min_sem_meteor': 0.12820512820512822},
            meteor_eval(model_output=model_outs, references=references)
        )

    def test_write_to_file(self):
        path = "out/eval/sem_meteor.csv"
        model_outs, references = build_synthetic_model_outputs()

        meteor_evaluator = MeteorEvaluator(save_to_file=True)
        meteor_evaluator(model_output=model_outs, references=references)

        self.assertTrue(os.path.exists(path))
        file_contents_df = pd.read_csv(path)
        self.assertListEqual(list1=file_contents_df.columns.to_list(),
                             list2=["model_out", "reference", "sem_meteor"])
