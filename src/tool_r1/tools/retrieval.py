"""Immutable, deterministic BM25 index using only the standard library."""
from __future__ import annotations
from collections import Counter, defaultdict
import heapq
import math
import re


def tokenize(text: str) -> list[str]:
    # Keep entity IDs intact so NOVA-123 differs from NOVA-123-WEST.
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]", text.lower())


class BM25:
    def __init__(self, documents, k1: float = 1.5, b: float = .75):
        self.documents = []
        self.postings = defaultdict(list)
        self.lengths = []
        self.k1, self.b = k1, b
        ids = set()
        for index, document in enumerate(documents):
            doc_id, text = document["id"], document["text"]
            if not isinstance(doc_id, str) or not isinstance(text, str) or doc_id in ids:
                raise ValueError("Corpus requires unique string IDs and text")
            ids.add(doc_id)
            self.documents.append({"id": doc_id, "text": text})
            tokens = tokenize(text)
            self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term].append((index, count))
        self.average = sum(self.lengths) / max(1, len(self.lengths)) or 1.0

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must have 1..2000 characters")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be an integer between 1 and 5")
        scores = defaultdict(float)
        count = len(self.documents)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, ())
            inverse = math.log(1 + (count - len(postings) + .5) / (len(postings) + .5))
            for index, frequency in postings:
                norm = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                scores[index] += inverse * frequency * (self.k1 + 1) / norm
        indices = heapq.nsmallest(top_k, scores, key=lambda i: (-scores[i], self.documents[i]["id"]))
        return [dict(self.documents[i], score=round(scores[i], 8)) for i in indices]

