import sys
import json
import math
import re
from typing import List, Tuple


class BM25:
    """BM25 text retrieval model."""
    def __init__(self) -> None:
        self.k1: float = 1.5
        self.b: float = 0.75
        self.corpus: list = []
        self.idf: dict = {}
        self.avgdl: float = 0.0
        self.STOPWORDS: set = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "be",
            "been",
            "being",
            "have",
            "has",
            "do",
            "does",
            "did",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "and",
            "or",
            "but",
            "if",
            "not",
            "this",
            "that",
            "it",
            "its",
            "what",
            "how",
            "which",
            "who",
            "when",
            "where",
            "why",
            "all",
            "each",
            "both",
            "more",
            "so",
            "than",
            "too",
            "very",
            "just",
            "no",
        }

    def tokenize(self, text: str) -> List[str]:
        """Convert text into tokens."""
        clean_text = re.sub(r"[^\w\s]", " ", text)
        words = clean_text.lower().split()
        tokens = [w for w in words if w not in self.STOPWORDS]
        if not tokens:
            return words
        return tokens

    def index(self, texts: List[str]) -> None:
        """Create the BM25 index."""
        self.corpus = [self.tokenize(text) for text in texts]
        self.avgdl = sum(len(doc) for doc in self.corpus) / len(self.corpus)
        counter = {}
        n = len(self.corpus)
        for doc in self.corpus:
            see = set()
            for word in doc:
                if word not in see:
                    if word not in counter:
                        counter[word] = 1
                    else:
                        counter[word] += 1
                    see.add(word)
        for term, df in counter.items():
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    def calculate_bm25(self, tf: float, idf: float, dl: int) -> float:
        """Calculate BM25 score."""
        up = tf * (self.k1 + 1)
        down = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        bm25 = idf * (up / down)
        return bm25

    def searcher(self, query: str, k: int) -> List[Tuple[float, int]]:
        """Search documents using BM25."""
        token_query = self.tokenize(query)
        scores = []
        i = 0
        for doc in self.corpus:
            score: float = 0.0
            for word in token_query:
                tf = doc.count(word)
                idf = self.idf.get(word, 0)
                score += self.calculate_bm25(tf, idf, len(doc))
            scores.append((score, i))
            i += 1
        result = sorted(scores, reverse=True)
        return result[:k]

    def save(self, path: str) -> None:
        """Save BM25 data to a JSON file."""
        data = {"idf": self.idf, "avgdl": self.avgdl, "corpus": self.corpus}
        with open(path, "w", encoding="utf-8", errors="ignore") as file:
            json.dump(data, file, indent=2)

    def load(self, path: str) -> None:
        """Load BM25 data from a JSON file."""
        try:
            with open(path, "r") as file:
                content = json.load(file)
            self.idf = content["idf"]
            self.avgdl = content["avgdl"]
            self.corpus = content["corpus"]
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            sys.exit(1)
