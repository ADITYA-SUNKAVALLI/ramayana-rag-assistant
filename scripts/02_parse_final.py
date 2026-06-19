#!/usr/bin/env python3
"""
scripts/02_parse_final.py

Parses cleaned_text/full_corpus.txt (the single merged document covering the
whole Ramayana Q&A book) into cleaned_text/all_chapters.jsonl, one JSON
record per chapter, using the simplified "resolved answer only" schema:

{
  "chunk_id": "bala_kanda_ch01",
  "kanda": "బాల కాండము",
  "sarga_range": [1,2,3,4,5,6],
  "chapter_ordinal": "ఒకటవ అధ్యాయము",
  "chapter_number": 1,
  "qa_pairs": [
    {"q_no": 1, "question": "...", "answer": "..."},
    ...
  ],
  "source_text_excerpt_chars": [start, end]
}

Run:
    python scripts/02_parse_final.py

Inputs:
    cleaned_text/full_corpus.txt   (merged corpus, built by 01b_merge_batches.py)

Outputs:
    cleaned_text/all_chapters.jsonl
    logs/parse_failures.md
    logs/parse_summary.md
"""

import re
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "cleaned_text" / "full_corpus.txt"
OUTPUT_PATH = BASE_DIR / "cleaned_text" / "all_chapters.jsonl"
FAILURES_LOG = BASE_DIR / "logs" / "parse_failures.md"
SUMMARY_LOG = BASE_DIR / "logs" / "parse_summary.md"

EXPECTED_TOTAL_CHAPTERS = 99  # from the table of contents

# ---------------------------------------------------------------------------
# Noise patterns to strip from anywhere inside answer text
# (running headers that repeat at every page break)
# ---------------------------------------------------------------------------
NOISE_LINE_PATTERNS = [
    r"^శ్రీరామాయణము\s*[-–]\s*ప్రశ్నావళి\s*$",
    r"^[^\n]{0,40}కాం?ండము\s*[-–]\s*[^\n]{0,30}అధ్యాయము\s*$",
    r"^\s*\d{1,4}\s*$",                 # bare page-number footer
    r"^\s*[ivxlcdm]{1,6}\s*$",          # bare roman numeral footer
    r"^\s*\.{3,}\s*$",                  # "......." divider line
]
NOISE_LINE_RE = re.compile(
    "(" + ")|(".join(NOISE_LINE_PATTERNS) + ")", re.IGNORECASE
)

PLACEHOLDER_RE = re.compile(r"జవాబుల\s*కాలంలో\s*ఉ(ంది|న్నది)\.?")

QUESTION_MARK_RE = re.compile(r"(\d+)\s*\.?\s*వ?\s*(?:ప్ర|ప)శ్న\s*:?")
ANSWER_MARK_RE = re.compile(r"జవాబు(?!లు)\s*:?")
ANSWER_KEY_MARK_RE = re.compile(r"జవాబులు\s*:?")

# chapter-start anchor: a "...సర్గలు" line followed (within a handful of
# lines) by a "...అధ్యాయము" line followed by a "1. ప్రశ్న" line.
SARGA_LINE_RE = re.compile(r"([\d,\s]+?)\s*సర్గలు\s*$")
ORDINAL_LINE_RE = re.compile(r"^(?!.*సర్గలు)([^\n]*అధ్యాయము)\s*$")
KANDA_NAME_RE = re.compile(r"కాం?ండము")

# end-of-chapter answer key, always exactly 3 entries for q1, q2, q3
KEY_BODY_RE = re.compile(
    r"1\s*\.\s*(\d*)\s*\.?\s*(.*?)\s*"
    r"2\s*\.\s*(\d*)\s*\.?\s*(.*?)\s*"
    r"3\s*\.\s*(.*)",
    re.DOTALL,
)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def strip_noise(text: str, extra_exact_lines=None) -> str:
    """Remove running-header / footer noise lines and rejoin the surrounding
    text into a single continuous paragraph, as if the page break never
    happened.

    extra_exact_lines: an optional set of normalized (whitespace-collapsed)
    strings that should also be treated as noise if a line matches them
    exactly — used for chapter-specific repeats like a standalone chapter
    ordinal line ("ఒకటవ అధ్యాయము") that shows up again right before the
    answer key.
    """
    extra_exact_lines = extra_exact_lines or set()
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if NOISE_LINE_RE.match(stripped):
            continue
        if _normalize_ws(stripped) in extra_exact_lines:
            continue
        kept.append(stripped)
    # join with single spaces, collapse any resulting multi-space runs
    joined = " ".join(l for l in kept if l)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def find_chapter_anchors(lines):
    """Return list of (anchor_line_index, kanda, sarga_range, ordinal)."""
    anchors = []
    n = len(lines)
    for i, line in enumerate(lines):
        m = SARGA_LINE_RE.search(line.strip())
        if not m:
            continue
        sarga_digits = re.findall(r"\d+", m.group(1))
        if not sarga_digits:
            continue
        sarga_range = [int(d) for d in sarga_digits]

        # look forward up to 4 lines for the ordinal line
        ordinal = None
        ordinal_idx = None
        for j in range(i, min(i + 5, n)):
            om = ORDINAL_LINE_RE.match(lines[j].strip())
            if om:
                ordinal = om.group(1).strip()
                ordinal_idx = j
                break
        if ordinal is None:
            continue

        # confirm a "1. ప్రశ్న" line follows within a few lines
        confirmed = False
        for k in range(ordinal_idx, min(ordinal_idx + 4, n)):
            qm = QUESTION_MARK_RE.search(lines[k])
            if qm and qm.group(1) == "1":
                confirmed = True
                break
        if not confirmed:
            continue

        # look backward up to 3 lines (including current) for kanda name
        kanda = None
        for j in range(i, max(i - 4, -1), -1):
            if KANDA_NAME_RE.search(lines[j]):
                kanda = re.sub(r"^\s*\d+\s*\.\s*", "", lines[j].strip())
                kanda = re.sub(r"[\d,\s]+సర్గలు.*$", "", kanda).strip()
                break
        if kanda is None:
            kanda = "UNKNOWN_KANDA"

        anchors.append({
            "start_line": i,
            "kanda": kanda,
            "sarga_range": sarga_range,
            "chapter_ordinal": ordinal,
        })
    return anchors


def parse_chapter_body(body: str, kanda: str, chapter_ordinal: str):
    """Parse the Q&A body (after the ordinal line, before the next chapter
    anchor) into qa_pairs + raw key text. Returns (qa_pairs_raw, key_text)
    or raises ValueError with a reason."""

    extra_noise = {
        _normalize_ws(chapter_ordinal),
        _normalize_ws(kanda),
        _normalize_ws(f"{kanda} - {chapter_ordinal}"),
        _normalize_ws(f"{kanda} – {chapter_ordinal}"),
    }

    # split off the answer-key block
    key_match = ANSWER_KEY_MARK_RE.search(body)
    if key_match:
        qa_section = body[: key_match.start()]
        key_text = body[key_match.end():]
    else:
        # Fallback: some chapters omit the "జవాబులు" label and go
        # straight into the "1.X. ... 2.X. ... 3. ..." key pattern.
        fallback_match = re.search(r"(?:^|\n)\s*1\s*\.\s*\d\s*\.", body)
        if not fallback_match:
            raise ValueError("no జవాబులు (answer key) block found")
        qa_section = body[: fallback_match.start()]
        key_text = body[fallback_match.start():]

    # cut key_text at the first "......." divider or end of string
    div = re.search(r"\.{3,}", key_text)
    if div:
        key_text = key_text[: div.start()]

    # find all question markers in the qa_section
    q_matches = list(QUESTION_MARK_RE.finditer(qa_section))
    if len(q_matches) != 5:
        raise ValueError(
            f"expected 5 question markers, found {len(q_matches)}"
        )

    qa_pairs_raw = []
    for idx, qm in enumerate(q_matches):
        q_no = int(qm.group(1))
        seg_start = qm.end()
        seg_end = q_matches[idx + 1].start() if idx + 1 < len(q_matches) else len(qa_section)
        segment = qa_section[seg_start:seg_end]

        am = ANSWER_MARK_RE.search(segment)
        if am:
            question_raw = segment[: am.start()]
            answer_raw = segment[am.end():]
        else:
            # No జవాబు label — options listed directly after the
            # question. Split on the question mark instead.
            qmark = segment.find("?")
            if qmark == -1:
                raise ValueError(f"no జవాబు marker or '?' found for q{q_no}")
            question_raw = segment[: qmark + 1]
            answer_raw = segment[qmark + 1:]

        question = strip_noise(question_raw, extra_noise)
        question = question.rstrip(":").strip()

        is_placeholder = bool(PLACEHOLDER_RE.search(answer_raw))
        answer_clean = strip_noise(answer_raw, extra_noise)

        qa_pairs_raw.append({
            "q_no": q_no,
            "question": question,
            "raw_answer": answer_clean,
            "is_placeholder": is_placeholder,
        })

    return qa_pairs_raw, key_text, extra_noise


def parse_answer_key(key_text: str, extra_noise=None):
    """Parse the 3-entry answer key block. Returns dict q_no -> resolved text,
    or raises ValueError."""
    cleaned = strip_noise(key_text, extra_noise)
    m = KEY_BODY_RE.search(cleaned)
    if not m:
        raise ValueError("answer key body did not match expected 3-entry pattern")
    opt1, ans1, opt2, ans2, ans3 = m.groups()
    return {
        1: ans1.strip(),
        2: ans2.strip(),
        3: ans3.strip(),
    }


def build_chunk_id(chapter_number: int) -> str:
    """Sequential, ASCII-safe identifier. Kept simple and stable — Telugu
    text is preserved in the 'kanda' field itself, not encoded into the id.
    A nicer English-slug id (e.g. 'bala_kanda_ch01') can be assigned later
    during the translation phase, once kanda names are in English."""
    return f"chapter_{chapter_number:03d}"


def main():
    if not CORPUS_PATH.exists():
        raise SystemExit(f"ERROR: corpus file not found at {CORPUS_PATH}")

    raw_text = CORPUS_PATH.read_text(encoding="utf-8")
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw_text.split("\n")

    anchors = find_chapter_anchors(lines)

    # locate true start of content: first anchor whose kanda line isn't
    # part of the table of contents (TOC rows never reach this function
    # since they lack a following "1. ప్రశ్న" within range, already filtered
    # by find_chapter_anchors's confirmation step)
    if not anchors:
        raise SystemExit("ERROR: no chapter anchors found at all — check corpus format")

    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    failures = []
    chapters_out = []
    chapter_number = 0

    for idx, anchor in enumerate(anchors):
        chapter_number += 1
        start_line = anchor["start_line"]
        end_line = anchors[idx + 1]["start_line"] if idx + 1 < len(anchors) else len(lines)
        chapter_text = "\n".join(lines[start_line:end_line])

        # body = everything after the ordinal line's first occurrence
        ordinal_pos = chapter_text.find(anchor["chapter_ordinal"])
        body = chapter_text[ordinal_pos + len(anchor["chapter_ordinal"]):] if ordinal_pos != -1 else chapter_text

        try:
            qa_pairs_raw, key_text, extra_noise = parse_chapter_body(
                body, anchor["kanda"], anchor["chapter_ordinal"]
            )
            key_map = parse_answer_key(key_text, extra_noise)

            qa_pairs = []
            for qa in qa_pairs_raw:
                q_no = qa["q_no"]
                if q_no in key_map and key_map[q_no]:
                    # MCQ (q1/q2) or resolved short-answer (q3): use key text
                    answer = key_map[q_no]
                else:
                    answer = qa["raw_answer"]

                if not qa["question"]:
                    raise ValueError(f"q{q_no} has empty question")

                needs_review = (not answer) or bool(PLACEHOLDER_RE.search(answer))

                qa_entry = {
                    "q_no": q_no,
                    "question": qa["question"],
                    "answer": None if needs_review else answer,
                }
                if needs_review:
                    qa_entry["needs_review"] = True
                qa_pairs.append(qa_entry)

            if len(qa_pairs) != 5:
                raise ValueError(f"expected 5 qa_pairs, got {len(qa_pairs)}")

            record = {
                "chunk_id": build_chunk_id(chapter_number),
                "kanda": anchor["kanda"],
                "sarga_range": anchor["sarga_range"],
                "chapter_ordinal": anchor["chapter_ordinal"],
                "chapter_number": chapter_number,
                "qa_pairs": qa_pairs,
            }
            chapters_out.append(record)

        except ValueError as e:
            failures.append({
                "chapter_number": chapter_number,
                "kanda": anchor["kanda"],
                "chapter_ordinal": anchor["chapter_ordinal"],
                "reason": str(e),
                "raw_text": chapter_text[:4000],
            })

    # write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in chapters_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # write failures log
    with open(FAILURES_LOG, "w", encoding="utf-8") as f:
        f.write("# Parsing Failures\n\n")
        if not failures:
            f.write("No failures.\n")
        for fail in failures:
            f.write(
                f"**Chapter {fail['chapter_number']} "
                f"({fail['kanda']} - {fail['chapter_ordinal']})** failed: "
                f"{fail['reason']}\n\n"
            )
            f.write("```\n" + fail["raw_text"] + "\n```\n\n")

    # write summary log
    total_found = len(anchors)
    total_ok = len(chapters_out)
    total_failed = len(failures)
    with open(SUMMARY_LOG, "w", encoding="utf-8") as f:
        f.write("# Parse Summary\n\n")
        f.write(f"- Total chapters found: {total_found}\n")
        f.write(f"- Successfully parsed: {total_ok}\n")
        f.write(f"- Flagged for review: {total_failed}\n")
        if total_found != EXPECTED_TOTAL_CHAPTERS:
            f.write(
                f"\n**WARNING:** expected {EXPECTED_TOTAL_CHAPTERS} chapters "
                f"(from table of contents) but found {total_found}. "
                f"Check chapter-boundary detection before trusting output.\n"
            )

    print(f"Total chapters found: {total_found}")
    print(f"Successfully parsed:  {total_ok}")
    print(f"Flagged for review:   {total_failed}")
    if total_found != EXPECTED_TOTAL_CHAPTERS:
        print(
            f"WARNING: expected {EXPECTED_TOTAL_CHAPTERS} chapters, "
            f"found {total_found}. See {SUMMARY_LOG}"
        )
    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"Failures log:       {FAILURES_LOG}")
    print(f"Summary log:        {SUMMARY_LOG}")


if __name__ == "__main__":
    main()