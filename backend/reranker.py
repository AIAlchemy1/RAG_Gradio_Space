# backend/reranker.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reranker:
    def __init__(self, model_name: str):
        """Initialize Reranker with cross-encoder model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # ✅ Evaluation mode

    def rerank(self, query: str, documents: list[str]) -> list[str]:
        """Rerank documents based on their relevance to the query."""
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Tokenize pairs
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Predict relevance scores
        with torch.no_grad():
            scores = self.model(**encoded).logits.squeeze(-1)

        # Sort documents by scores (higher = better)
        ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        return ranked_docs
