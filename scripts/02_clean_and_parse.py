#!/usr/bin/env python3
"""
scripts/02_parse_final.py – Final chapter parser for the merged Ramayana corpus.
Produces cleaned_text/all_chapters.jsonl.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ─── Config ───────────────────────────────────────────────────────────────
CORPUS_PATH = Path("cleaned_text/full_corpus.txt")
PAGE_INDEX_PATH = Path("cleaned_text/page_index.json")
OUTPUT_DIR = Path("cleaned_text")
LOG_DIR = Path("logs")
OUTPUT_JSONL = OUTPUT_DIR / "all_chapters.jsonl"
FAILURE_LOG = LOG_DIR / "parse_failures.md"
PARSE_SUMMARY_LOG = LOG_DIR / "parse_summary.md"

# ─── Patterns ─────────────────────────────────────────────────────────────
SARGA_LINE = re.compile(r'\d+.*సర్గలు', re.UNICODE)                # e.g. "1,2,3,4,5,6 సర్గలు"
ORDINAL_LINE = re.compile(r'అధ్యాయము$')                           # line ending with అధ్యాయము
QA_START = re.compile(r'^\s*(\d{1,2})\s*[.।]\s*ప్రశ్న\s*:?\s*(.*)', re.UNICODE)
ANSWER_HEADER = re.compile(r'^\s*జవాబు\s*:?\s*(.*)', re.UNICODE)
OPTION_LINE = re.compile(r'^\s*\d\s*[.)]\s*(.*)')
KEY_LINE = re.compile(r'^\s*జవాబులు\s*:?\s*(.*)', re.UNICODE)
PLACEHOLDER_PAT = re.compile(r'జవాబుల\s+కాలంలో\s+ఉన్[దన]ి\.?')   # both ఉంది/ఉన్నది
PHRASE_NOISE = re.compile(r'శ్రీరామాయణము\s*[-–—]\s*ప్రశ్నావళి')   # the recurring phrase
RUNNING_HEADER = re.compile(r'(?=.*కాండ)(?=.*అధ్యాయము)')          # line contains both
BARE_PAGE_NUMBER = re.compile(r'^\s*\d{1,3}\s*$')                    # isolated page number

# Kanda name → English slug mapping
KANDA_SLUG = {
    'బాలకాండ': 'bala_kanda',
    'బాల కాండ': 'bala_kanda',
    'అయోధ్యకాండ': 'ayodhya_kanda',
    'అయోధ్య కాండ': 'ayodhya_kanda',
    'అరణ్యకాండ': 'aranya_kanda',
    'అరణ్య కాండ': 'aranya_kanda',
    'కిష్కింధకాండ': 'kishkindha_kanda',
    'కిష్కింధ కాండ': 'kishkindha_kanda',
    'సుందరకాండ': 'sundara_kanda',
    'సుందర కాండ': 'sundara_kanda',
    'యుద్ధకాండ': 'yuddha_kanda',
    'యుద్ధ కాండ': 'yuddha_kanda',
    'ఉత్తరకాండ': 'uttara_kanda',
    'ఉత్తర కాండ': 'uttara_kanda',
}
def kanda_to_slug(telugu_name: str) -> str:
    # Normalise spaces and try to match
    name = telugu_name.strip().replace('ము', '')
    for key, slug in KANDA_SLUG.items():
        if key in name or name in key:
            return slug
    return "unknown_kanda"

# ─── Helper functions ─────────────────────────────────────────────────────
def load_corpus_and_index():
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    page_numbers = []
    if PAGE_INDEX_PATH.exists():
        with open(PAGE_INDEX_PATH, 'r', encoding='utf-8') as f:
            page_numbers = json.load(f)
        if len(page_numbers) != len(lines):
            print("Warning: page_index length mismatch. Ignoring page index.")
            page_numbers = []
    return lines, page_numbers

def is_blank(line: str) -> bool:
    return not line.strip()

def is_noise_line(line: str) -> bool:
    """Return True if the line should be treated as structural noise and removed."""
    stripped = line.strip()
    if not stripped:
        return True
    if BARE_PAGE_NUMBER.match(stripped):
        return True
    if PHRASE_NOISE.search(stripped):
        return True
    if RUNNING_HEADER.search(stripped):
        return True
    # Also catch leftover "--- Page N ---" markers (just in case)
    if re.match(r'^--- Page \d+ ---\s*$', stripped):
        return True
    return False

def extract_sarga_info(line: str) -> Tuple[str, List[int]]:
    """From a sarga line, extract the kanda name (if present) and the sarga numbers."""
    # The line may look like "37, 38, 39, 40, 41, 42 సర్గలు" or "బాలకాండము 1,2,3,4,5,6 సర్గలు"
    kanda = ""
    sarga_numbers = []
    # Try to find "సర్గలు" preceded by digits and optionally a kanda name
    m = re.match(r'^(.*?)([\d,\s]+)\s*సర్గలు', line)
    if m:
        prefix = m.group(1).strip()
        # prefix could be kanda name, or empty
        if prefix and re.search(r'కాండ', prefix):
            kanda = prefix
        nums_str = m.group(2)
        sarga_numbers = [int(n) for n in re.findall(r'\d+', nums_str)]
    else:
        # fallback: split at last sequence of digits/comma before "సర్గలు"
        parts = line.split('సర్గలు')[0].strip()
        # find all numbers
        sarga_numbers = [int(n) for n in re.findall(r'\d+', parts)]
        # try to extract kanda name: everything before the first digit
        m2 = re.match(r'^(.*?)\d', parts)
        if m2:
            kanda = m2.group(1).strip()
    return kanda, sarga_numbers

def parse_key_entries(key_text: str) -> List[Dict]:
    """Parse the answer key lines and return a list of entries: {q_no, option_no, answer}."""
    entries = []
    for line in key_text.strip().splitlines():
        # Split line on multiple spaces
        parts = re.split(r'\s{2,}', line.strip())
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m_mcq = re.match(r'(\d)\.(\d)\.\s*(.*)', part)
            if m_mcq:
                entries.append({
                    "q_no": int(m_mcq.group(1)),
                    "option_no": int(m_mcq.group(2)),
                    "answer": m_mcq.group(3).strip()
                })
                continue
            m_short = re.match(r'(\d)\.\s*(.*)', part)
            if m_short:
                entries.append({
                    "q_no": int(m_short.group(1)),
                    "option_no": None,
                    "answer": m_short.group(2).strip()
                })
    return entries

# ─── Chapter boundary detection ──────────────────────────────────────────
def find_chapter_starts(lines: List[str]) -> List[int]:
    """
    Return line indices where a real chapter starts.
    A chapter start = a sarga line (digits + "సర్గలు") followed within a few lines
    by an ordinal line (ending "అధ్యాయము") and then a Q1 line.
    """
    starts = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if SARGA_LINE.search(line) and 'సర్గలు' in line:
            # Look ahead for ordinal within 8 lines
            found_ordinal = False
            j = i + 1
            while j < len(lines) and j - i <= 8:
                if ORDINAL_LINE.search(lines[j]):
                    found_ordinal = True
                    break
                j += 1
            if not found_ordinal:
                i += 1
                continue
            # Look for Q1 within next 8 lines after ordinal
            k = j + 1
            while k < len(lines) and k - j <= 8:
                qm = QA_START.match(lines[k])
                if qm and qm.group(1) == '1':
                    starts.append(i)
                    # Skip ahead to Q1 line to avoid overlapping matches
                    i = k
                    break
                k += 1
            else:
                i += 1
        else:
            i += 1
    return starts

# ─── Chapter parser ──────────────────────────────────────────────────────
def parse_chapter_block(block_lines: List[str], kanda_override: str = None) -> Optional[Dict]:
    """
    Parse a chapter block. Returns a chapter dict or None on failure.
    kanda_override can be provided if the sarga line didn't contain the kanda name
    (in which case we'll try to extract it from a nearby line).
    """
    # Step 1: extract header info from the first few lines
    # The block starts with the sarga line (already validated). We'll gather kanda,
    # sarga_range, and ordinal from the following lines.
    idx = 0
    sarga_line = block_lines[0]
    extracted_kanda, sarga_range = extract_sarga_info(sarga_line)
    if not sarga_range:
        return None  # malformed

    # Find the ordinal line (ends with "అధ్యాయము")
    ordinal = ""
    while idx < len(block_lines):
        if ORDINAL_LINE.search(block_lines[idx]):
            ordinal = block_lines[idx].strip()
            break
        idx += 1
    if not ordinal:
        # try to find it within the first 10 lines as fallback
        for i in range(min(10, len(block_lines))):
            if ORDINAL_LINE.search(block_lines[i]):
                ordinal = block_lines[i].strip()
                break
    if not ordinal:
        return None

    # Determine the final kanda name. If the sarga line didn't contain one,
    # look at the line just before the sarga line in the original corpus (not available here)
    # or check if there's a kanda line earlier in the block.
    kanda = extracted_kanda if extracted_kanda else ""
    if not kanda:
        # Search backwards from the sarga line (but we only have the block; the block starts at sarga line).
        # Usually the kanda name is on the line immediately before the sarga line in the merged text.
        # Since the block starts at the sarga line, we'll check the line right before it in the full corpus later.
        # We'll handle this after we have the full lines list. For now, set to empty and we'll patch later.
        pass

    # ─── Parse Q&A pairs and the key ───────────────────────────────────────
    # We'll walk through the block from start, collecting content until the end.
    # We'll maintain a current "mode": searching for Q numbers.
    qa_pairs_raw = []   # list of {q_no, question, answer, is_placeholder?, options?}
    placeholder_q_no = None
    current_q = None
    question_text = ""
    answer_text = ""
    collecting_answer = False
    expecting_options = False
    options_list = []
    key_entries = []

    # We'll iterate over lines after we have passed the header lines (sarga, ordinal).
    # First, advance idx past the ordinal line we found.
    while idx < len(block_lines) and ORDINAL_LINE.search(block_lines[idx]) is None:
        idx += 1
    idx += 1  # move past ordinal

    def flush_current_qa():
        nonlocal current_q, question_text, answer_text, expecting_options, options_list, placeholder_q_no
        if current_q and question_text:
            if current_q in (1, 2) and expecting_options:
                # MCQ not finalized until key; we'll store options and placeholder flag
                # Actually we don't know answer yet; we'll record the question and options.
                qa_pairs_raw.append({
                    "q_no": current_q,
                    "type": "mcq",
                    "question": question_text.strip(),
                    "options": options_list[:],
                    "answer": None,
                    "is_placeholder": False
                })
            elif current_q == 3:
                # Check if answer is placeholder
                is_ph = bool(PLACEHOLDER_PAT.search(answer_text))
                if is_ph:
                    placeholder_q_no = current_q
                qa_pairs_raw.append({
                    "q_no": current_q,
                    "type": "short_answer",
                    "question": question_text.strip(),
                    "answer": answer_text.strip(),
                    "is_placeholder": is_ph
                })
            else:  # 4,5
                qa_pairs_raw.append({
                    "q_no": current_q,
                    "type": "descriptive",
                    "question": question_text.strip(),
                    "answer": answer_text.strip(),
                    "is_placeholder": False
                })
        current_q = None
        question_text = ""
        answer_text = ""
        expecting_options = False
        options_list = []

    # Walk through the block lines
    while idx < len(block_lines):
        raw_line = block_lines[idx]
        line = raw_line.strip()

        # Skip blank or noise lines while collecting answer? We'll skip noise lines always.
        if is_noise_line(raw_line):
            idx += 1
            continue

        # Check for start of a new Q&A pair
        qm = QA_START.match(raw_line)
        if qm:
            # Flush previous QA
            flush_current_qa()
            current_q = int(qm.group(1))
            question_text = qm.group(2).strip()
            collecting_answer = False
            expecting_options = False
            answer_text = ""
            options_list = []
            idx += 1
            continue

        # Check for answer header
        am = ANSWER_HEADER.match(raw_line)
        if am and current_q:
            # Entering answer section
            collecting_answer = True
            answer_text = am.group(1).strip()
            if current_q in (1, 2):
                expecting_options = True
                options_list = []
            idx += 1
            continue

        # If we are collecting answer and we encounter a key line, stop collecting.
        if KEY_LINE.match(raw_line):
            flush_current_qa()
            # Now parse the key
            key_text = raw_line[len("జవాబులు"):].strip()  # after the marker
            # collect continuation lines
            idx += 1
            while idx < len(block_lines):
                if QA_START.match(block_lines[idx]) or is_noise_line(block_lines[idx]) or ORDINAL_LINE.search(block_lines[idx]) or (SARGA_LINE.search(block_lines[idx]) and 'సర్గలు' in block_lines[idx]):
                    # stop if we hit a new chapter header or next question
                    break
                key_text += "\n" + block_lines[idx].strip()
                idx += 1
            key_entries = parse_key_entries(key_text)
            # We'll process key after loop
            continue

        # If expecting options (MCQ), collect them
        if expecting_options and current_q in (1, 2):
            # Look for option lines: digit, dot or bracket, then text
            om = OPTION_LINE.match(raw_line)
            if om and len(options_list) < 4:
                options_list.append(om.group(1).strip())
                idx += 1
                continue
            else:
                # If we get something that isn't an option, stop expecting options.
                expecting_options = False

        # If collecting answer, append the line (after stripping noise? noise already removed)
        if collecting_answer and not expecting_options:
            # Append with a space to build paragraph
            if answer_text:
                answer_text += " " + line
            else:
                answer_text = line
            idx += 1
            continue

        # Default: move to next line
        idx += 1

    # Flush last QA
    flush_current_qa()

    # ─── Process answer key and finalize answers ─────────────────────────
    # Identify which q_no is the placeholder one (should be exactly one)
    placeholder_q = placeholder_q_no
    # If not set, maybe default to 3, but we'll try to deduce from key
    if placeholder_q is None:
        # find any qa_pairs_raw with answer containing placeholder (in case we missed)
        for qa in qa_pairs_raw:
            if PLACEHOLDER_PAT.search(qa.get("answer", "")):
                placeholder_q = qa["q_no"]
                qa["is_placeholder"] = True
                break
    # If still None, assume 3 (but that would be suspicious; we'll log later)
    if placeholder_q is None:
        placeholder_q = 3  # most common

    # Map key entries to qa_pairs
    for ke in key_entries:
        q = ke["q_no"]
        # Find the corresponding raw qa pair
        for qa in qa_pairs_raw:
            if qa["q_no"] == q:
                if qa["type"] == "mcq":
                    qa["answer"] = ke["answer"]  # will override later if needed, but we want the correct answer text
                elif qa["type"] == "short_answer" and qa["is_placeholder"]:
                    qa["answer"] = ke["answer"]  # replace placeholder with real answer
                # Descriptive q's are not in key (they have inline answers)

     # Build final qa_pairs list with only {q_no, question, answer}
    final_qa = []
    for qa in qa_pairs_raw:
        qno = qa["q_no"]
        question = qa["question"]
        answer = qa.get("answer")          # may be None if key missing

        # For MCQs, answer must be resolved from key
        if qa["type"] == "mcq":
            if answer is None:
                return None                # unresolved → chapter invalid

        # Sanity: answer must not contain the placeholder phrase
        if answer is not None and PLACEHOLDER_PAT.search(answer):
            return None

        # Fallback to empty string (shouldn't normally happen)
        answer = answer if answer is not None else ""

        final_qa.append({
            "q_no": qno,
            "question": question,
            "answer": answer.strip()
        })

    if len(final_qa) != 5:
        return None
    # ─── Build chapter dict ──────────────────────────────────────────────
    # slug
    slug = kanda_to_slug(kanda) if kanda else "unknown"
    # chapter_number will be assigned later by caller
    chapter_dict = {
        "chunk_id": "",   # filled later
        "kanda": kanda,
        "sarga_range": sarga_range,
        "chapter_ordinal": ordinal,
        "chapter_number": 0,  # set by caller
        "qa_pairs": final_qa,
        "source_pages": []
    }
    return chapter_dict

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged corpus...")
    lines, page_numbers = load_corpus_and_index()
    print(f"Corpus: {len(lines)} lines.")

    # Find chapter start indices
    starts = find_chapter_starts(lines)
    print(f"Found {len(starts)} chapter start(s).")

    if not starts:
        print("No chapters found. Exiting.")
        return

    # Save front matter (lines before first start)
    first_start = starts[0]
    front_matter = lines[:first_start]
    front_path = OUTPUT_DIR / "front_matter.txt"
    with open(front_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(front_matter))
    print(f"Front matter saved to {front_path} ({len(front_matter)} lines).")

    # Build chapter blocks
    chapter_blocks = []
    for i, start in enumerate(starts):
        end = starts[i+1] if i+1 < len(starts) else len(lines)
        chapter_blocks.append((start, end, lines[start:end]))

    successful = []
    failures = []
    # We may need to look at the line just before the sarga line for kanda name
    for ch_idx, (start_line, end_line, block) in enumerate(chapter_blocks, 1):
        # Try to get kanda name from the line immediately before the block (in original lines)
        if start_line > 0:
            prev_line = lines[start_line-1].strip()
            if re.search(r'కాండ', prev_line):
                # It might be a standalone kanda header line; use it as override
                parsed = parse_chapter_block(block, kanda_override=prev_line)
                if parsed is None:
                    # Try without override
                    parsed = parse_chapter_block(block)
            else:
                parsed = parse_chapter_block(block)
        else:
            parsed = parse_chapter_block(block)

        if parsed:
            # Fill chapter_number and chunk_id
            parsed["chapter_number"] = ch_idx
            kanda_name = parsed["kanda"]
            if not kanda_name:
                # Try to extract from block header lines (e.g., first line of block might contain kanda)
                # We'll scan first few lines for a word containing "కాండ"
                for line in block[:10]:
                    m = re.search(r'(.*కాండ(ము)?)', line)
                    if m:
                        kanda_name = m.group(1).strip()
                        parsed["kanda"] = kanda_name
                        break
            slug = kanda_to_slug(kanda_name) if kanda_name else "unknown"
            parsed["chunk_id"] = f"{slug}_ch{ch_idx:02d}"

            # Source pages
            if page_numbers and start_line < len(page_numbers) and (end_line-1) < len(page_numbers):
                p_start = page_numbers[start_line] if page_numbers[start_line] != 0 else None
                p_end = page_numbers[end_line-1] if page_numbers[end_line-1] != 0 else None
                parsed["source_pages"] = [p_start, p_end]

            successful.append(parsed)
        else:
            failures.append((ch_idx, block))

    # Write JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        for ch in successful:
            fout.write(json.dumps(ch, ensure_ascii=False) + '\n')
    print(f"\n{len(successful)} chapters written to {OUTPUT_JSONL}")

    # Write failures
    if failures:
        with open(FAILURE_LOG, 'w', encoding='utf-8') as flog:
            flog.write("# Parse Failures\n\n")
            for ch_idx, block in failures:
                flog.write(f"**Chapter {ch_idx}** (block starting at line {starts[ch_idx-1]}) failed.\n\n")
                flog.write(f"```\n{''.join(block[:2000])}\n```\n\n")
        print(f"{len(failures)} chapter(s) failed (see {FAILURE_LOG}).")
    else:
        print("All chapters parsed successfully.")

    # Summary
    total_found = len(starts)
    total_parsed = len(successful)
    total_failed = len(failures)
    warning = "" if total_found == 99 else f"WARNING: Expected 99 chapters but found {total_found}. Possible boundary-detection bug."

    with open(PARSE_SUMMARY_LOG, 'w', encoding='utf-8') as sm:
        sm.write("# Parse Summary\n\n")
        sm.write(f"- Total chapters found: {total_found}\n")
        sm.write(f"- Successfully parsed: {total_parsed}\n")
        sm.write(f"- Flagged to parse_failures.md: {total_failed}\n")
        if warning:
            sm.write(f"- {warning}\n")
    print(f"Summary written to {PARSE_SUMMARY_LOG}")
    print(warning)

if __name__ == "__main__":
    main()