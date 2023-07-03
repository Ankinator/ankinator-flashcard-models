from typing import List, Tuple, Dict
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    Abstract class to implement different evaluators
    """

    def __call__(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:
        return self.evaluate(model_output=model_output, references=references)

    @abstractmethod
    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:
        """
        Evaluate model outputs on the gold standard with a set of metrics, covering language quality, Semantic Similarity and Correctness
        :param model_output: Output from the model to be evaluated; Follows the convention from our celery pipeline:
            List of tuples: one tuple per page number:
                int: pagenumber,
                List[str]: multiple generated questions if applicable
        :param references: References from the gold standard to compute metrics
        :return: A dictionary with metrics as keys and the matric values as values
        """
        pass