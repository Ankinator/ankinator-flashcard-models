import os.path
import unittest
from src.evaluation.semantic_similarity import SentenceTransformerEvaluator
import pandas as pd


class TestSemanticSim(unittest.TestCase):

    @staticmethod
    def _build_synthetic_model_outputs():
        model_outs = [
            (1, ["What is the capital of France?"]),
            (2, ["How do plants obtain energy?"]),
            (3, ["Are dogs considered mammals?"]),
            (4, ["What are the symptoms of COVID-19?"]),
            (5, ["Why is exercise important for a healthy lifestyle?"])
        ]
        refences = [
            (1, ["Which city is the capital of France?"]),
            (2, ["What is the source of energy for plants?"]),
            (3, ["Do cats fall under the category of mammals?"]),
            (4, ["Can you list the signs of COVID-19?"]),
            (5, ["What are the benefits of incorporating exercise into a daily routine?"])
        ]
        return model_outs, refences

    def test_semantic_smiliarity_calculation(self):
        model_outs, references = self._build_synthetic_model_outputs()

        sentence_transformer_evaluator = SentenceTransformerEvaluator(save_to_file=False)
        self.assertDictEqual(
            {'avg_cos_sim': 0.8464279174804688, 'max_cos_sim': 0.9416357278823853, 'min_cos_sim': 0.702666699886322},
            sentence_transformer_evaluator(model_output=model_outs, references=references)
        )

    def test_write_to_file(self):
        path = "out/eval/cosine_sim.csv"
        model_outs, references = self._build_synthetic_model_outputs()

        sentence_transformer_evaluator = SentenceTransformerEvaluator(save_to_file=True)
        sentence_transformer_evaluator(model_output=model_outs, references=references)

        self.assertTrue(os.path.exists(path))
        file_contents_df = pd.read_csv(path)
        self.assertListEqual(list1=file_contents_df.columns.to_list(),
                             list2=["model_out", "reference", "similiarities"])

    def test_plot(self):
        path = "out/eval/sim.png"
        model_outs, references = self._build_synthetic_model_outputs()

        sentence_transformer_evaluator = SentenceTransformerEvaluator(save_to_file=False)
        sentence_transformer_evaluator(model_output=model_outs, references=references)
        sentence_transformer_evaluator.plot()
        self.assertTrue(os.path.exists(path))
