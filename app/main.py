from flask import Flask

from embeder import collection, embedding_model
from rag_chain import ask_qwen

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    answer = ask_qwen(question)
    return jsonify({"answer": answer})

from flask import request, jsonify

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("file")

    if not uploaded_file:
        return jsonify({"error": "No file provided"}), 400

    if not uploaded_file.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files are supported"}), 400

    filename = uploaded_file.filename
    content = uploaded_file.read().decode("utf-8")

    chunks = [content[i:i+512] for i in range(0, len(content), 512)]
    existing_ids = set(collection.get()['ids'])

    new_chunks = 0
    for idx, chunk in enumerate(chunks):
        id_ = f"{filename}-{idx}"
        if id_ not in existing_ids:
            emb = embedding_model.encode(chunk).tolist()
            collection.add(documents=[chunk], embeddings=[emb], ids=[id_])
            new_chunks += 1

    return jsonify({"message": f"{new_chunks} chunks added from {filename}."})

@app.route("/vectors", methods=["GET"])
def get_all_vectors():
    try:
        results = collection.get()

        if not results:
            return jsonify({"error": "collection.get() returned None or empty"}), 404
        documents = results.get("documents")
        ids = results.get("ids")
        embeddings = results.get("embeddings")

        if not isinstance(documents, list) or not isinstance(ids, list) or not isinstance(embeddings, list):
            return jsonify({
                "error": "Invalid result format from collection.get()",
                "details": {
                    "documents": str(type(documents)),
                    "ids": str(type(ids)),
                    "embeddings": str(type(embeddings))
                }
            }), 500

        response = []
        for i in range(len(ids)):
            doc = documents[i] if i < len(documents) else None
            emb = embeddings[i] if i < len(embeddings) else None

            if emb is None:
                return jsonify({"error": f"Embedding at index {i} is None"}), 500

            response.append({
                "id": ids[i],
                "document": doc,
                "embedding_preview": emb[:5],
                "embedding_size": len(emb)
            })

        return jsonify({
            "total": len(response),
            "vectors": response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Ready to answer!")
    app.run(host="0.0.0.0", port=5000)
