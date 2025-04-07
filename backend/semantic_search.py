# backend/semantic_search.py

import os
import gradio as gr
import lancedb
from sentence_transformers import SentenceTransformer

from backend.reranker import Reranker

# ✅ Environment Variables
DB_PATH = os.getenv("DB_PATH", ".lancedb")
TABLE_NAME = os.getenv("TABLE_NAME", "default_table")
VECTOR_COLUMN = os.getenv("VECTOR_COLUMN", "vector")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "text")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ✅ Connect to LanceDB
try:
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
except Exception as e:
    raise RuntimeError(f"❌ Could not connect to LanceDB or open table '{TABLE_NAME}': {e}")

# ✅ Load embedding model and reranker
retriever = SentenceTransformer(EMB_MODEL)
reranker = Reranker(RERANKER_MODEL)

def retrieve(query: str, top_k: int):
    """Retrieve documents from LanceDB and rerank them using Cross-Encoder."""
    try:
        # Encode query into vector
        query_vec = retriever.encode(query)

        # Search database
        documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN).limit(top_k).to_list()
        documents_text = [doc.get(TEXT_COLUMN, "") for doc in documents]

        if not documents_text:
            return ["⚠️ No documents found for the query."]

        # Rerank based on relevance
        reranked_documents = reranker.rerank(query, documents_text)

        return reranked_documents

    except Exception as e:
        raise gr.Error(f"❌ Retrieval failed: {str(e)}")
