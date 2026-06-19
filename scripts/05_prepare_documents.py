import json
from pathlib import Path

INPUT_FILE = Path("translated/rag_final_chapters.jsonl")
OUTPUT_FILE = Path("translated/documents.jsonl")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

documents = []

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        chapter = json.loads(line)

        text_parts = []

        text_parts.append(
            f"Kanda: {chapter['kanda']}"
        )

        text_parts.append(
            f"Chapter: {chapter['chapter_ordinal']}"
        )

        text_parts.append(
            f"Chapter Number: {chapter['chapter_number']}"
        )

        text_parts.append("")

        for qa in chapter["qa_pairs"]:

            text_parts.append(
                f"Question {qa['q_no']}:"
            )

            text_parts.append(
                qa["question"]
            )

            text_parts.append("")

            text_parts.append(
                "Answer:"
            )

            text_parts.append(
                qa["answer"]
            )

            text_parts.append("")
            text_parts.append("-" * 50)
            text_parts.append("")

        searchable_text = "\n".join(text_parts)

        document = {
            "id": chapter["chunk_id"],
            "text": searchable_text,
            "metadata": {
                "kanda": chapter["kanda"],
                "chapter_ordinal": chapter["chapter_ordinal"],
                "chapter_number": chapter["chapter_number"],
                "sarga_range": chapter["sarga_range"]
            }
        }

        documents.append(document)

        count += 1

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for doc in documents:

        f.write(
            json.dumps(
                doc,
                ensure_ascii=False
            ) + "\n"
        )

print(
    f"Documents Created: {count}"
)

print(
    f"Saved To: {OUTPUT_FILE}"
)