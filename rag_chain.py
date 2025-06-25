import os
import ollama
from embeder import get_context

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=ollama_host)

def ask_qwen(question: str) -> str:
    context = get_context(question)

    if not context or context.strip() == "":
        return (
            "I'm sorry, I can only assist with questions related to "
            "company services and operations."
        )

    context = context.strip()[:1000]

    prompt = (
        "You identify as Alliance GPT, you help assist the employees and customers "
        "of Alliance Pro, which is the company.\n\n"
        "Answer the question using the context below. If the question is not related "
        "to the provided context, politely decline to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    response = ollama_client.chat(
        model="mistral:7b",
        messages=[{"role": "user", "content": prompt}],
      )

    return response["message"]["content"]