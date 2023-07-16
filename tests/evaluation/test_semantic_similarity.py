import os.path
import unittest
from src.evaluation.metrics.semantic_similarity import SentenceTransformerEvaluator
import pandas as pd

from tests.evaluation import build_synthetic_model_outputs


class TestSemanticSim(unittest.TestCase):

    def test_semantic_smiliarity_calculation(self):
        model_outs, references = build_synthetic_model_outputs()

        sentence_transformer_evaluator = SentenceTransformerEvaluator(save_to_file=False)
        expected_result = {
            'avg_cos_sim': 0.8464279174804688,
            'max_cos_sim': 0.9416357278823853,
            'min_cos_sim': 0.702666699886322
        }

        actual_result = sentence_transformer_evaluator(model_output=model_outs, references=references)

        for key, expected_value in expected_result.items():
            actual_value = actual_result[key]
            expected_rounded = round(expected_value, 6)
            actual_rounded = round(actual_value, 6)

            self.assertEqual(expected_rounded, actual_rounded)

    def test_write_to_file(self):
        path = "out/eval/cosine_sim.csv"
        model_outs, references = build_synthetic_model_outputs()

        sentence_transformer_evaluator = SentenceTransformerEvaluator(save_to_file=True)
        sentence_transformer_evaluator(model_output=model_outs, references=references)

        self.assertTrue(os.path.exists(path))
        file_contents_df = pd.read_csv(path)
        self.assertListEqual(list1=file_contents_df.columns.to_list(),
                             list2=["model_out", "reference", "similiarities"])

    def test_plot(self):
        path = "out/eval/sim.png"
        model_outs, references = build_synthetic_model_outputs()

        sentence_transformer_evaluator = SentenceTransformerEvaluator(save_to_file=False)
        sentence_transformer_evaluator(model_output=model_outs, references=references)
        sentence_transformer_evaluator.plot()
        self.assertTrue(os.path.exists(path))
