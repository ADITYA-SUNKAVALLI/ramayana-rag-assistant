import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = Path("translated/documents.jsonl")

EMBEDDINGS_FILE = Path("embeddings/embeddings.npy")
METADATA_FILE = Path("embeddings/metadata.json")

EMBEDDINGS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

documents = []
metadata = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        doc = json.loads(line)

        documents.append(
            doc["text"]
        )

        metadata.append(
            {
                "id": doc["id"],
                **doc["metadata"]
            }
        )

print(
    f"Documents Loaded: {len(documents)}"
)

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(
    EMBEDDINGS_FILE,
    embeddings
)

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print(
    f"Embeddings Shape: {embeddings.shape}"
)

print(
    f"Saved: {EMBEDDINGS_FILE}"
)

print(
    f"Saved: {METADATA_FILE}"
)