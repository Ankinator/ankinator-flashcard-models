from typing import List, Tuple, Dict

from sentence_transformers import SentenceTransformer, util

from src.evaluation.evaluator import Evaluator
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))


class SentenceTransformerEvaluator(Evaluator):

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def evaluate(self, model_output: List[Tuple[int, List[str]]], references: List[Tuple[int, List[str]]]) -> Dict[
        str, float]:

        def extract_strings(inp_collection: List[Tuple[int, List[str]]]):
            """
                Select the first string from each tuple in the input collection, as evaluation example.
            """
            return [t[1][0] for t in inp_collection]

        sentences_from_model = extract_strings(model_output)
        sentences_from_reference = extract_strings(references)

        model_sentence_embeddings = self.model.encode(sentences=sentences_from_model, convert_to_tensor=True)
        reference_sentence_embeddings = self.model.encode(sentences=sentences_from_reference, convert_to_tensor=True)

        similiarities =




