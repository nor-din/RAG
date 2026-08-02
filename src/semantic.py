import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

class Semantic:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None

    def index(self, texts):
        self.embeddings = self.model.encode(texts)

    def search(self, query, k):
        vector_query = self.model.encode(query)
        score = util.cos_sim(vector_query, self.embeddings)
        np_resulte = score.numpy()
        result = np.argsort(np_resulte[0])[::-1][:k]
        return result.tolist()

    def save(self, path):
        tensor = torch.from_numpy(self.embeddings)
        torch.save(tensor, path)

    def load(self, path):
        tensor = torch.load(path)
        self.embeddings = tensor.numpy()
