from flask import Flask, request, jsonify

from embeder import load_and_embed_docs
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

if __name__ == "__main__":
    print("Embedding documents...")
    load_and_embed_docs("../documents")
    print("Ready to answer!")
    app.run(host="0.0.0.0", port=5000)
