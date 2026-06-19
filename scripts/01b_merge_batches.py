#!/usr/bin/env python3
"""
scripts/01b_merge_batches.py – Merge all extracted batch files into one
continuous corpus, remove page markers, build a line-level page index,
and remove duplicate lines at batch boundaries.
"""

import re
import json
from pathlib import Path
from typing import List, Tuple

# ─── Configuration ───────────────────────────────────────────────────────
INPUT_DIR = Path("extracted_text")
OUTPUT_DIR = Path("cleaned_text")
LOG_DIR = Path("logs")
CORPUS_FILE = OUTPUT_DIR / "full_corpus.txt"
PAGE_INDEX_FILE = OUTPUT_DIR / "page_index.json"
LOG_FILE = LOG_DIR / "merge_log.md"

PAGE_MARKER = re.compile(r'^--- Page (\d+) ---\s*$')

# ─── Helpers ─────────────────────────────────────────────────────────────

def parse_filename_page_range(filename: str) -> Tuple[int, int]:
    """
    Extract (start_page, end_page) from a filename like
    batch_001_pages_1-18.txt or batch_014_pages_235-248.txt.
    """
    m = re.search(r'pages_(\d+)-(\d+)', filename)
    if not m:
        raise ValueError(f"Could not parse page range from {filename}")
    return int(m.group(1)), int(m.group(2))

def sort_batch_files(files: List[Path]) -> List[Path]:
    """Sort batch files by their start page number."""
    return sorted(files, key=lambda f: parse_filename_page_range(f.name)[0])

def read_batch_lines_with_page(batch_path: Path) -> List[Tuple[str, int]]:
    """
    Read a batch file, return a list of (line_text, page_number).
    Page markers are removed; lines that are bare page numbers (common OCR footer)
    are also skipped.
    """
    lines_with_page = []
    current_page = None
    with open(batch_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            # Check for page marker
            marker = PAGE_MARKER.match(line)
            if marker:
                current_page = int(marker.group(1))
                continue
            # Skip lines that look like a bare page number (e.g., "18" or " 18")
            stripped = line.strip()
            if stripped.isdigit() and 1 <= len(stripped) <= 4:
                # Could be a page number footer – skip it unless it's part of real text
                # We'll also skip if it's a single number on its own
                # To be safe, skip if it matches a page number pattern and is isolated
                # (no surrounding text that would indicate it's not a footer)
                # But we can't be 100% sure; in Ramayana, a standalone number is almost
                # certainly a page number. So skip.
                continue
            # Record line with the page it came from (if known)
            page = current_page if current_page else 0  # 0 = unknown
            lines_with_page.append((line, page))
    return lines_with_page

def remove_boundary_overlaps(
    merged_lines: List[Tuple[str, int]],
    new_lines: List[Tuple[str, int]],
    lookback: int = 8
) -> List[Tuple[str, int]]:
    """
    Detect if the tail of merged_lines overlaps with the head of new_lines.
    If overlapping lines are found (up to `lookback` lines), skip the
    duplicate lines in new_lines and return the truncated new_lines.
    Also returns a log message if overlap was removed.
    """
    # We'll compare line texts (ignore page numbers for the comparison)
    for overlap_len in range(min(lookback, len(merged_lines), len(new_lines)), 0, -1):
        tail_texts = [t for t, _ in merged_lines[-overlap_len:]]
        head_texts = [t for t, _ in new_lines[:overlap_len]]
        if tail_texts == head_texts:
            # Found overlap of `overlap_len` lines
            truncated = new_lines[overlap_len:]
            return truncated, overlap_len
    return new_lines, 0

# ─── Main function ───────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = list(INPUT_DIR.glob("batch_*.txt"))
    if not batch_files:
        print("No batch_*.txt files found in extracted_text/. Run 01_extract.py first.")
        return

    # Sort by start page
    sorted_files = sort_batch_files(batch_files)
    print(f"Found {len(sorted_files)} batch files.")

    merged_lines: List[Tuple[str, int]] = []
    total_overlaps_removed = 0
    total_pages_seen = set()
    log_entries = []

    for bf in sorted_files:
        print(f"Processing {bf.name} ...")
        lines_with_page = read_batch_lines_with_page(bf)

        # Track unique pages
        pages_in_batch = {pg for _, pg in lines_with_page if pg != 0}
        total_pages_seen.update(pages_in_batch)

        if merged_lines:
            # Check for overlap at boundary
            new_lines, overlap = remove_boundary_overlaps(merged_lines, lines_with_page)
            if overlap > 0:
                print(f"  -> Removed {overlap} duplicate line(s) at boundary.")
                total_overlaps_removed += overlap
                log_entries.append(
                    f"- `{bf.name}`: {overlap} duplicate line(s) removed at start."
                )
        else:
            new_lines = lines_with_page

        merged_lines.extend(new_lines)

    # Verify we have content
    if not merged_lines:
        print("No text lines merged. Check batch files.")
        return

    # Write full_corpus.txt (only the text lines, no page numbers)
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        for text, _ in merged_lines:
            f.write(text + '\n')
    print(f"\nFull corpus written to {CORPUS_FILE} ({len(merged_lines)} lines).")

    # Build page index: a list the same length as merged_lines,
    # each entry is the source page number (or 0 if unknown).
    page_index = [pg for _, pg in merged_lines]
    with open(PAGE_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(page_index, f, ensure_ascii=False)
    print(f"Page index written to {PAGE_INDEX_FILE}.")

    # Count distinct pages
    distinct_pages = len(total_pages_seen)
    print(f"Distinct pages merged: {distinct_pages}")

    # Write log
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        log.write("# Merge Log\n\n")
        log.write(f"- **Batch files processed**: {len(sorted_files)}\n")
        log.write(f"- **Distinct pages merged**: {distinct_pages}\n")
        log.write(f"- **Total lines in corpus**: {len(merged_lines)}\n")
        log.write(f"- **Overlapping lines removed**: {total_overlaps_removed}\n\n")
        if log_entries:
            log.write("## Boundary Overlaps Detected\n\n")
            for entry in log_entries:
                log.write(entry + '\n')
        else:
            log.write("No boundary overlaps were detected.\n")
    print(f"Merge log written to {LOG_FILE}.")

if __name__ == "__main__":
    main()