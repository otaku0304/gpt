import os
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Create a persistent Chroma client using the new style
chroma_client = PersistentClient(path="./chroma_store")  # path must exist or be writable
collection = chroma_client.get_or_create_collection("alliance_docs")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_context(question, top_k=1):
    question_emb = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_emb], n_results=top_k)
    return "\n---\n".join(results['documents'][0]) if results['documents'] else ""
  