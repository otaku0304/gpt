import os
from sentence_transformers import SentenceTransformer
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("alliance_docs")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def get_context(question, top_k=1):
    question_emb = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_emb], n_results=top_k)
    return "\n---\n".join(results['documents'][0]) if results['documents'] else ""
