import os
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

chroma_client = PersistentClient(path="/root/.chromadb")
collection = chroma_client.get_or_create_collection("alliance_docs")

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def load_and_embed_docs(folder="documents"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, folder)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    existing_ids = set(collection.get()['ids'])
    print("Existing IDs:", existing_ids)

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                for idx, chunk in enumerate(chunks):
                    id_ = f"{filename}-{idx}"
                    if id_ not in existing_ids:
                        emb = embedding_model.encode(chunk).tolist()
                        collection.add(
                            documents=[chunk],
                            embeddings=[emb],
                            ids=[id_]
                        )

def get_context(question, top_k=3):
    question_emb = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]  # avoid crash on empty
    return "\n---\n".join(docs) if docs else ""

