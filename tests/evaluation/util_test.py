import unittest

from tests.evaluation import build_synthetic_model_outputs
from src.evaluation.util import extract_strings, add_dimension_for_processing


class TestUtils(unittest.TestCase):

    def test_extract_strings(self):
        model_outs, references = build_synthetic_model_outputs()
        strings = extract_strings(model_outs)
        self.assertListEqual(
            [
                "What is the capital of France?",
                "How do plants obtain energy?",
                "Are dogs considered mammals?",
                "What are the symptoms of COVID-19?",
                "Why is exercise important for a healthy lifestyle?"
            ], strings
        )

    def test_add_dimension(self):
        ref = [
            ["Why is exercise important for a healthy lifestyle?"],
            ["How do plants obtain energy?"]
        ]
        inputs = [
            "Why is exercise important for a healthy lifestyle?",
            "How do plants obtain energy?"
        ]

        self.assertListEqual(
            ref,
            add_dimension_for_processing(inp_collection=inputs)
        )
