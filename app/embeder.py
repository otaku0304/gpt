import os
from sentence_transformers import SentenceTransformer
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("alliance_docs")

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def load_and_embed_docs(folder="documents"):
    # Make the path absolute relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, folder)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f" Folder not found: {folder_path}")

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r") as f:
                text = f.read()
                chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                for idx, chunk in enumerate(chunks):
                    emb = embedding_model.encode(chunk).tolist()
                    collection.add(
                        documents=[chunk],
                        embeddings=[emb],
                        ids=[f"{filename}-{idx}"]
                    )

def get_context(question, top_k=1):
    question_emb = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_emb], n_results=top_k)
    return "\n---\n".join(results['documents'][0]) if results['documents'] else ""
