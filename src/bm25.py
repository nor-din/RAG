import math
import json

class BM25:
    def __init__(self):
        self.k1 = 1.5
        self.b = 0.75
        self.corpus = []
        self.idf = {}
        self.avgdl = 0

    def index(self, texts):
        self.corpus = [word.lower().split() for word in texts]
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

    def calculate_bm25(self, tf, idf, dl):
        up = tf * (self.k1 + 1)
        down = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        bm25 = idf * (up / down)
        return bm25

    def searcher(self,query, k):
        token_query = [word.lower() for word in query.split()]
        scores = []
        i = 0
        for doc in self.corpus:
            score = 0
            for word in token_query:
                tf = doc.count(word)
                idf = self.idf.get(word, 0)
                score += self.calculate_bm25(tf, idf, len(doc))
            scores.append((score, i))
            i += 1
        result = sorted(scores, reverse=True)
        return result[:k]

    def save(self, path):
        data = {
            "idf":self.idf,
            "avgdf":self.avgdl,
            "corpus":self.corpus
        }
        with open(path, 'w', encoding="utf-8", errors="ignore") as file:
            json.dump(data, file, indent=2)

    def load(self, path):
        with open(path, 'r') as file:
            content = json.load(file)
        self.idf = content['idf']
        self.avgdl = content['avgdf']
        self.corpus = content['corpus']
