import os
import json
import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Loading FAISS index...")

index = faiss.read_index(
    "vectorstore/ramayana.index"
)

documents = []

with open(
    "translated/documents.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        documents.append(
            json.loads(line)
        )

print(
    f"Loaded {len(documents)} documents"
)

SYSTEM_PROMPT = """
You are a Ramayana expert assistant.

Answer ONLY from the provided context.

Rules:

1. Give a complete sentence answer.
2. Explain briefly when context allows.
3. Never use knowledge outside the retrieved context.
4. If the answer is not present, say:
   'I could not find this information in the Ramayana knowledge base.'
5. Do not mention document IDs or chapter IDs.
6. Keep answers concise but informative.
"""

while True:

    question = input(
        "\nAsk Question (or type exit): "
    )

    if question.lower() == "exit":
        break

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        k=3
    )

    context_parts = []

    for idx in indices[0]:

        doc = documents[idx]

        context_parts.append(
            doc["text"]
        )

    context = "\n\n".join(
        context_parts
    )

    user_prompt = f"""
Context:

{context}

Question:

{question}

Answer:
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    print("\n")
    print("=" * 80)
    print(answer)
    print("=" * 80)