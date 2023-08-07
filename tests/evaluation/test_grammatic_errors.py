import os
import unittest

import pandas as pd

from src.evaluation.metrics.grammatic_errors import LanguageToolEvaluator


class TestLanguageTool(unittest.TestCase):

    def test_error_calculation(self):

        model_out = [(1, ["A text of texts of texts"]), (2, ["A text with a error insode"])]
        language_tool_eval = LanguageToolEvaluator(save_to_file=False)
        self.assertDictEqual(
            {'GRAMMAR': 1, 'MISC': 1},
            language_tool_eval(model_output=model_out, references=model_out)
        )

    def test_write_to_file(self):
        path = "out/eval/lt_errors.csv"
        model_out = [(1, ["A text of texts of texts"]), (2, ["A text with a error insode"])]

        language_tool_eval = LanguageToolEvaluator(save_to_file=True)
        language_tool_eval(model_output=model_out, references=model_out)

        self.assertTrue(os.path.exists(path))
        file_contents_df = pd.read_csv(path)
        self.assertListEqual(list1=file_contents_df.columns.to_list(),
                             list2=['Unnamed: 0', 'GRAMMAR', 'MISC'])
