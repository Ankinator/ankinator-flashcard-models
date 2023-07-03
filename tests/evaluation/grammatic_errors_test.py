import unittest

from src.evaluation.metrics.grammatic_errors import LanguageToolEvaluator


class TestLanguageTool(unittest.TestCase):

    def test_error_calculation(self):

        model_out = [(1, ["A text of texts of texts"]), (2, ["A text with a error insode"])]
        language_tool_eval = LanguageToolEvaluator(save_to_file=False)
        self.assertDictEqual(
            {'GRAMMAR': 1, 'MISC': 1},
            language_tool_eval(model_output=model_out, references=model_out)
        )
