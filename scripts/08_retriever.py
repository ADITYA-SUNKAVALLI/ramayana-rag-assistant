import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vectorstore/ramayana.index"
METADATA_PATH = "vectorstore/metadata.json"
DOCUMENTS_PATH = "translated/documents.jsonl"

print("Loading model...")
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

documents = []

with open(
    DOCUMENTS_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        documents.append(
            json.loads(line)
        )

while True:

    query = input(
        "\nAsk Question (or type exit): "
    )

    if query.lower() == "exit":
        break

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        k=3
    )

    print("\nTop Results:\n")

    for rank, idx in enumerate(indices[0], start=1):

        doc = documents[idx]

        print("=" * 80)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {scores[0][rank-1]:.4f}"
        )

        print(
            f"Document ID: {doc['id']}"
        )

        print(
            f"Kanda: {doc['metadata']['kanda']}"
        )

        print(
            f"Chapter: {doc['metadata']['chapter_ordinal']}"
        )

        print()

        print(
            doc["text"][:1000]
        )

        print("\n")