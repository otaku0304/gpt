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
You are Alliance GPT, an AI assistant designed to provide direct and helpful responses using the given context.

Answer the following question **briefly but clearly**. If the question is unrelated to the context, politely decline to answer. Remember when the user wishes you, again wish him politely.Also please member to memorize the previous chats in the session.

### Context:
{context}

### Question:
{question}  

### Answer (Concise, to-the-point):
"""

    response = ollama_client.chat(
        model="llama3:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant. Always give short and relevant answers using the provided context only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response['message']['content']
