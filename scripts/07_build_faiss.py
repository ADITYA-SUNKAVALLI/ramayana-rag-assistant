import json
import faiss
import numpy as np
from pathlib import Path

EMBEDDINGS_FILE = Path(
    "embeddings/embeddings.npy"
)

METADATA_FILE = Path(
    "embeddings/metadata.json"
)

INDEX_FILE = Path(
    "vectorstore/ramayana.index"
)

INDEX_METADATA_FILE = Path(
    "vectorstore/metadata.json"
)

INDEX_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    f"Embeddings Shape: {embeddings.shape}"
)

dimension = embeddings.shape[1]

print(
    f"Vector Dimension: {dimension}"
)

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings.astype("float32")
)

faiss.write_index(
    index,
    str(INDEX_FILE)
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

with open(
    INDEX_METADATA_FILE,
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
    f"Vectors Indexed: {index.ntotal}"
)

print(
    f"Index Saved: {INDEX_FILE}"
)

print(
    f"Metadata Saved: {INDEX_METADATA_FILE}"
)