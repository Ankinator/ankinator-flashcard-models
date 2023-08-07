import os
from typing import List, Tuple, Dict

import pandas as pd
import language_tool_python

from collections import Counter, defaultdict
from itertools import chain

from src.evaluation.metrics.evaluator import Evaluator
from src.evaluation.util import extract_strings


class LanguageToolEvaluator(Evaluator):

    def __init__(self, save_to_file=True):
        super().__init__(save_to_file)
        self.errors: List[language_tool_python.Match] = []
        self.error_statistics: Counter = Counter()
        self.lang_tool = language_tool_python.LanguageTool('en')

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:

        self.sentences_from_model = [l[0] for l in extract_strings(model_output)]
        self.sentences_from_reference = extract_strings(references)

        lt_matches = [self.lang_tool.check(t) for t in self.sentences_from_model]
        self.errors = list(chain.from_iterable(lt_matches))

        categories = [match.category for match in self.errors]
        self.error_statistics.update(categories)

        if self.save_to_file:
            self.save_scores_to_file()

        return dict(self.error_statistics)

    def save_scores_to_file(self, path="out/eval/lt_errors.csv"):
        if not os.path.exists(path):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        self.get_dataframe().to_csv(path_or_buf=path)

    def get_dataframe(self) -> pd.DataFrame:
        d: Dict[Counter] = defaultdict(Counter)
        for match in self.errors:
            d[match.context].update([match.category])

        return pd.DataFrame(d).transpose().fillna(0)
