import unittest

from tests.evaluation import build_synthetic_model_outputs
from src.evaluation.metrics.rouge import RougeScoreEvaluator


class TestRouge(unittest.TestCase):

    def test_rouge_calc(self):
        model_outs, references = build_synthetic_model_outputs()

        rouge_eval = RougeScoreEvaluator(save_to_file=False)
        self.assertDictEqual(
            {'rouge1_fmeasure': 0.39748987555503845, 'rouge1_precision': 0.46095237135887146,
             'rouge1_recall': 0.35422077775001526, 'rouge2_fmeasure': 0.2069930136203766,
             'rouge2_precision': 0.226666659116745, 'rouge2_recall': 0.1904762089252472,
             'rougeL_fmeasure': 0.36672061681747437, 'rougeL_precision': 0.42095237970352173,
             'rougeL_recall': 0.3292207717895508, 'rougeLsum_fmeasure': 0.36672061681747437,
             'rougeLsum_precision': 0.42095237970352173, 'rougeLsum_recall': 0.3292207717895508},
            rouge_eval(model_output=model_outs, references=references)
        )
