from typing import List, Tuple, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from src.evaluation.evaluator import Evaluator
import torch
import os
import matplotlib.pyplot as plt

from src.evaluation.util import extract_strings


class SentenceTransformerEvaluator(Evaluator):
    """
    Evaluator to compute cosine similarities with sentence transformers,
    all values are accessible from within the object
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', save_to_file=True):
        self.sentences_from_reference: List[str] = []
        self.sentences_from_model: List[str] = []
        self.similiarities = None
        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name)
        self.save_to_file = save_to_file

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:

        self.sentences_from_model = extract_strings(model_output)
        self.sentences_from_reference = extract_strings(references)

        model_sentence_embeddings = self.model.encode(sentences=self.sentences_from_model, convert_to_tensor=True)
        reference_sentence_embeddings = self.model.encode(sentences=self.sentences_from_reference,
                                                          convert_to_tensor=True)

        self.similiarities = util.cos_sim(model_sentence_embeddings, reference_sentence_embeddings)

        if self.save_to_file:
            self.save_scores_to_file()

        return {
            "avg_cos_sim": torch.diag(self.similiarities).mean().item(),
            "max_cos_sim": torch.diag(self.similiarities).max().item(),
            "min_cos_sim": torch.diag(self.similiarities).min().item()
        }

    def save_scores_to_file(self, path='out/eval/cosine_sim.csv'):
        """
        Saves the similarity values on the main diagonal of the similiarities tensor to a .csv file along with its
        text references
        :param path: path to the output file
        :return: None
        """

        if not os.path.exists(path):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        self.get_dataframe().to_csv(path_or_buf=path, index=False)

    def get_dataframe(self) -> pd.DataFrame:
        similiarities_diag = torch.diag(self.similiarities).numpy()
        return pd.DataFrame({
            "model_out": self.sentences_from_model,
            "reference": self.sentences_from_reference,
            "similiarities": similiarities_diag
        })

    def plot(self, path="out/eval/sim.png"):
        plt.boxplot(self.get_dataframe()["similiarities"])
        plt.title("Semantic Similiarity measured by Cosine Sim")
        plt.savefig(path)
