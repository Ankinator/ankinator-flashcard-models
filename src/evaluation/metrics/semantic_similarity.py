from typing import List, Tuple, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from src.evaluation.metrics.evaluator import Evaluator
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
        super().__init__(save_to_file=save_to_file)
        self.similiarities = torch.tensor([])
        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:
        self.sentences_from_model = extract_strings(model_output)
        self.sentences_from_reference = extract_strings(references)

        for model_out, refs in zip(self.sentences_from_model, self.sentences_from_reference):
            model_sentence_embeddings = self.model.encode(sentences=model_out, convert_to_tensor=True)
            reference_sentence_embeddings = self.model.encode(sentences=refs,
                                                              convert_to_tensor=True)

            sims: torch.Tensor = util.cos_sim(model_sentence_embeddings, reference_sentence_embeddings)
            sims = sims.max().unsqueeze(0)

            self.similiarities = torch.concat([
                self.similiarities,
                sims
            ])

        if self.save_to_file:
            self.save_scores_to_file()

        return {
            "avg_cos_sim": self.similiarities.mean().item(),
            "max_cos_sim": self.similiarities.max().item(),
            "min_cos_sim": self.similiarities.min().item()
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
        similiarities_diag = self.similiarities.numpy()
        return pd.DataFrame({
            "model_out": self.sentences_from_model,
            "reference": self.sentences_from_reference,
            "max_similarity": similiarities_diag
        })

    def plot(self, path="out/eval/sim.png"):
        plt.boxplot(self.get_dataframe()["similiarities"])
        plt.title("Semantic Similiarity measured by Cosine Sim")
        plt.savefig(path)
