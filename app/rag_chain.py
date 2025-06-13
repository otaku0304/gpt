import os
import ollama

from embeder import get_context

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=ollama_host)

def ask_qwen(question):
    context = get_context(question)

    if not context or context.strip() == "":
        return "I'm sorry, I can only assist with questions related to company services and operations."

    prompt = f"""
You are Alliance GPT, a helpful assistant for employees and customers.

Answer the question using the context below. If the question is not related to the provided context, politely decline to answer.

Context:
{context}

Question:
{question}
"""
    response = ollama_client.chat(
        model="mistral:7b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
