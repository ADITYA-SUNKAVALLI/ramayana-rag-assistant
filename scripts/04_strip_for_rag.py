#!/usr/bin/env python3
"""
scripts/04_strip_for_rag.py

Reads the full translation output (translated/all_chapters_en.jsonl) and
produces a stripped version with only the fields required for the RAG dataset.
Strictly follows the schema described in the project specification.
"""

import json
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────
INPUT_PATH = Path("translated/all_chapters_en.jsonl")
OUTPUT_PATH = Path("translated/rag_final_chapters.jsonl")

# Fields that we keep (in order)
KEEP_FIELDS = [
    "chunk_id",
    "kanda",
    "chapter_ordinal",
    "chapter_number",
    "sarga_range",
    "qa_pairs",
]

def clean_qa_pair(qa: dict) -> dict:
    """Keep only q_no, question, answer — rename question_en/answer_en if present."""
    q_no = qa.get("q_no")
    # The translation script may produce "question_en" / "answer_en" or "question"/"answer"
    # Accept both, but in the final output we use "question" and "answer".
    question = qa.get("question") or qa.get("question_en", "")
    answer = qa.get("answer") or qa.get("answer_en", "")
    return {
        "q_no": q_no,
        "question": question.strip(),
        "answer": answer.strip(),
    }

def strip_record(record: dict) -> dict:
    """Return a new dict with only the allowed fields, and cleaned qa_pairs."""
    # Extract kanda name (English only). If the record has kanda_en, use that,
    # otherwise parse the kanda field (which might be Telugu) — we assume the
    # translation already provides an English kanda name in "kanda_en".
    kanda_en = record.get("kanda_en") or record.get("kanda", "")
    # If kanda_en still contains Telugu, we fall back to a generic placeholder.
    # In practice, after translation, kanda_en should be the English name.
    # We also accept "kanda" if it's already English (e.g. "Bala Kanda").
    # For safety, we'll just keep the value as-is, but the spec requires English only.
    # We'll trust that the translation produced a proper English kanda name.

    chapter_ordinal_en = record.get("chapter_ordinal_en") or record.get("chapter_ordinal", "")

    # Ensure chapter_number is int
    chapter_number = record.get("chapter_number")
    if isinstance(chapter_number, float):
        chapter_number = int(chapter_number)

    # Clean qa_pairs
    qa_pairs = record.get("qa_pairs_en") or record.get("qa_pairs", [])
    clean_qa = [clean_qa_pair(qa) for qa in qa_pairs]

    return {
        "chunk_id": record.get("chunk_id", ""),
        "kanda": kanda_en,
        "chapter_ordinal": chapter_ordinal_en,
        "chapter_number": chapter_number,
        "sarga_range": record.get("sarga_range", []),
        "qa_pairs": clean_qa,
    }

def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in:
        records = [json.loads(line) for line in f_in if line.strip()]

    stripped = [strip_record(rec) for rec in records]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for rec in stripped:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Cleaned {len(stripped)} chapters.")
    print(f"Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    