from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

chroma_client = PersistentClient(path="/root/.chromadb")
collection = chroma_client.get_or_create_collection("alliance_docs")

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_context(question, top_k=3):
    question_emb = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]  # avoid crash on empty
    return "\n---\n".join(docs) if docs else ""

