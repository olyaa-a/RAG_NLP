import json
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
from tqdm import tqdm

# Tokenizes text into lowercase words without stop words
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Define stop words for tokenization
STOP_WORDS = {'a', 'the', 'and', 'is', 'in', 'it', 'on', 'at', 'to', 'of'}

# BM25-based retriever for keyword search
class BM25Retriever:
    def __init__(self, document_path):
        self.documents = self.load_documents(document_path)
        self.tokenized_docs = [self.tokenize(doc['text']) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    # Loads documents from a JSON file
    def load_documents(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Tokenizes and removes stop words
    def tokenize(self, text):
        tokens = simple_tokenize(text)
        return [token for token in tokens if token not in STOP_WORDS]

    # Searches for top_n relevant documents using BM25
    def search(self, query, top_n=3):
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.documents[i], scores[i]) for i in ranked_indices[:top_n]]

# Dense retriever using SentenceTransformer for semantic search
class DenseRetriever:
    def __init__(self, model_name, document_path, embedding_path=None):
        self.model = SentenceTransformer(model_name)
        self.documents = self.load_documents(document_path)
        self.embedding_path = embedding_path
        self.embeddings = self.load_or_compute_embeddings()

    # Loads documents from a JSON file
    def load_documents(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Computes embeddings for all documents
    def compute_embeddings(self):
        texts = [doc['text'] for doc in self.documents]
        embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)

    # Saves computed embeddings to a file
    def save_embeddings(self):
        with open(self.embedding_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

    # Loads embeddings from a file or computes them if missing
    def load_or_compute_embeddings(self):
        if self.embedding_path:
            try:
                with open(self.embedding_path, 'rb') as f:
                    embeddings = pickle.load(f)
                    if not isinstance(embeddings, torch.Tensor):
                        raise ValueError("Loaded embeddings are not a tensor")
                    print(f"Loaded embeddings from {self.embedding_path}")
                    return embeddings
            except (FileNotFoundError, EOFError, ValueError):
                print(f"Embeddings file not found, corrupted, or invalid. Computing embeddings...")

        self.embeddings = self.compute_embeddings()
        if self.embedding_path:
            self.save_embeddings()
        return self.embeddings

    # Searches for top_n relevant documents using semantic similarity
    def search(self, query, top_n=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        if not isinstance(self.embeddings, torch.Tensor):
            raise ValueError("Embeddings are not a tensor. Check computation or loading process.")
        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
        ranked_indices = similarities.argsort(descending=True)[:top_n]
        return [(self.documents[i], similarities[i].item()) for i in ranked_indices]

# Combines BM25 and Dense retrievers with weighted scoring
class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, bm25_weight=0.7, dense_weight=0.3):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    # Performs hybrid search and combines results
    def search(self, query, top_n=3, use_bm25=True, use_dense=True):
        results = []

        if use_bm25:
            bm25_results = self.bm25_retriever.search(query, top_n)
            results.extend([(doc, score, 'BM25') for doc, score in bm25_results])

        if use_dense:
            dense_results = self.dense_retriever.search(query, top_n)
            results.extend([(doc, score, 'Dense') for doc, score in dense_results])

        combined_results = {}
        for doc, score, method in results:
            doc_id = doc['text']
            if doc_id not in combined_results:
                combined_results[doc_id] = {'doc': doc, 'bm25': 0, 'dense': 0}
            if method == 'BM25':
                combined_results[doc_id]['bm25'] = score
            elif method == 'Dense':
                combined_results[doc_id]['dense'] = score

        ranked_results = sorted(
            combined_results.values(),
            key=lambda x: self.bm25_weight * x['bm25'] + self.dense_weight * x['dense'],
            reverse=True
        )

        return [(res['doc'], self.bm25_weight * res['bm25'] + self.dense_weight * res['dense']) for res in ranked_results[:top_n]]