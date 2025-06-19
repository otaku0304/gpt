from flask import Flask, request, jsonify

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


if __name__ == "__main__":
    print("Ready to answer!")
    app.run(host="0.0.0.0", port=5000)
