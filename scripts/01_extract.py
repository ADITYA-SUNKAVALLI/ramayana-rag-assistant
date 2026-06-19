#!/usr/bin/env python3
"""
scripts/01_extract.py – Extract and batch the Ramayana PDF

1. Open raw_pdf/ramayana.pdf with PyMuPDF.
2. Extract text page by page, keeping page numbers.
3. Detect Kanda / chapter headings (Telugu & English patterns).
4. Choose batch splitting: fixed-size (≈18 pages) vs Kanda boundaries –
   whichever gives more even chunk sizes.
5. Write each batch to extracted_text/batch_XXX_<label>.txt
6. Log extraction details to logs/extraction_log.md
7. Print a summary table.
"""

import re
import sys
from pathlib import Path
from collections import OrderedDict
import statistics

import fitz  # PyMuPDF

# ─── Configuration ───────────────────────────────────────────────
PDF_PATH = Path("raw_pdf/ramayana.pdf")
OUTPUT_DIR = Path("extracted_text")
LOG_PATH = Path("logs/extraction_log.md")
TARGET_PAGE_BATCH = 18           # desired number of pages per batch
MIN_TEXT_LEN_FOR_VALID_PAGE = 50 # chars below → possible scanned image

# ─── Patterns for Kanda / Chapter headings ───────────────────────
# Telugu words for Kanda / Sarga, English "Kanda", "Sarga", "Chapter"
# Also lines that begin with Telugu numerals (e.g., ౧. బాలకాండ)
HEADING_PATTERNS = [
    re.compile(r'(?:^|\s)(కాండ|kanda)\b', re.IGNORECASE),           # Kanda
    re.compile(r'(?:^|\s)(సర్గ|sarga)\b', re.IGNORECASE),            # Sarga
    re.compile(r'(?:^|\s)(chapter)\b', re.IGNORECASE),               # Chapter
    re.compile(r'^[\u0C66-\u0C6F]+\s', re.UNICODE),                 # Telugu numeral start
]

def is_heading_line(line: str) -> bool:
    """Return True if the line matches any heading pattern."""
    if not line.strip():
        return False
    for pat in HEADING_PATTERNS:
        if pat.search(line):
            return True
    return False

def extract_kanda_name(line: str) -> str:
    """
    Try to extract a short Kanda identifier from a heading line.
    Falls back to a cleaned version of the line.
    """
    # Common Kanda names (Telugu and English)
    kanda_map = {
        'బాలకాండ': 'BalaKanda',
        'బాల కాండ': 'BalaKanda',
        'bala kanda': 'BalaKanda',
        'అయోధ్యకాండ': 'AyodhyaKanda',
        'అయోధ్య కాండ': 'AyodhyaKanda',
        'ayodhya kanda': 'AyodhyaKanda',
        'అరణ్యకాండ': 'AranyaKanda',
        'అరణ్య కాండ': 'AranyaKanda',
        'aranya kanda': 'AranyaKanda',
        'కిష్కింధకాండ': 'KishkindhaKanda',
        'కిష్కింధ కాండ': 'KishkindhaKanda',
        'kishkindha kanda': 'KishkindhaKanda',
        'సుందరకాండ': 'SundaraKanda',
        'సుందర కాండ': 'SundaraKanda',
        'sundara kanda': 'SundaraKanda',
        'యుద్ధకాండ': 'YuddhaKanda',
        'యుద్ధ కాండ': 'YuddhaKanda',
        'yuddha kanda': 'YuddhaKanda',
        'ఉత్తరకాండ': 'UttaraKanda',
        'ఉత్తర కాండ': 'UttaraKanda',
        'uttara kanda': 'UttaraKanda',
    }
    lower = line.strip().lower()
    for key, val in kanda_map.items():
        if key in lower:
            return val
    # Fallback: return first 30 chars, sanitised
    safe = re.sub(r'[^\w\s]', '', line.strip())  # remove punctuation
    return safe[:30].replace(' ', '_')

# ─── Extraction ──────────────────────────────────────────────────
def extract_pages(pdf_path: Path):
    """Return list of (page_num, text) for every page."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True) or ""
        pages.append((idx, text))
    doc.close()
    return pages

# ─── Chunking logic ──────────────────────────────────────────────
def kanda_ranges(pages):
    """
    Analyse pages and return list of (start, end, kanda_name)
    for each detected Kanda segment.  start/end are 1‑based page numbers.
    """
    segments = []
    current_start = 1
    current_kanda = "Unknown"
    for pg_num, text in pages:
        # Check if this page contains a heading
        heading = None
        for line in text.splitlines():
            if is_heading_line(line):
                heading = line
                break
        if heading:
            # Close previous segment if not the first page
            if pg_num > 1:
                segments.append((current_start, pg_num - 1, current_kanda))
            current_start = pg_num
            current_kanda = extract_kanda_name(heading)
    # Final segment
    segments.append((current_start, len(pages), current_kanda))
    return segments

def fixed_size_batches(num_pages, target_size=TARGET_PAGE_BATCH):
    """Return list of (start, end) page ranges for fixed-size batches."""
    batches = []
    for start in range(1, num_pages + 1, target_size):
        end = min(start + target_size - 1, num_pages)
        batches.append((start, end))
    return batches

def batch_sizes_from_ranges(ranges):
    return [(end - start + 1) for start, end in ranges]

def choose_best_batches(pages, segments):
    """
    Decide between pure page-based splitting and Kanda‑based splitting.
    Returns list of (start, end, label) batches.
    Label is a Kanda name or 'pages_X-Y'.
    """
    # 1. Page‑based batches
    page_batches = fixed_size_batches(len(pages))
    page_sizes = batch_sizes_from_ranges(page_batches)
    page_labels = [f"pages_{s}-{e}" for s, e in page_batches]

    # 2. Kanda‑based batches (each segment is one batch)
    kanda_batches = [(s, e, name) for s, e, name in segments]
    kanda_sizes = [(e - s + 1) for s, e, _ in kanda_batches]

    # Compute standard deviation (skip if only one batch)
    std_page = statistics.pstdev(page_sizes) if len(page_sizes) > 1 else 0
    std_kanda = statistics.pstdev(kanda_sizes) if len(kanda_sizes) > 1 else 0

    # If no Kanda detected, fallback to page‑based
    if not kanda_batches or all(name == "Unknown" for _,_,name in kanda_batches):
        print("No Kanda headings detected. Using fixed-size page batches.")
        return [(s, e, l) for (s,e), l in zip(page_batches, page_labels)]

    # Compare evenness – lower stddev is more even
    if std_kanda <= std_page:
        print(f"Kanda‑based splitting chosen (std dev {std_kanda:.1f} vs {std_page:.1f}).")
        return kanda_batches
    else:
        print(f"Page‑based splitting chosen (std dev {std_page:.1f} vs {std_kanda:.1f}).")
        return [(s, e, l) for (s,e), l in zip(page_batches, page_labels)]

# ─── Write output files ──────────────────────────────────────────
def write_batches(pages, batches):
    """Write one .txt file per batch into OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, (start, end, label) in enumerate(batches, start=1):
        fname = f"batch_{idx:03d}_{label}.txt"
        filepath = OUTPUT_DIR / fname
        with open(filepath, "w", encoding="utf-8") as f:
            for pg_num in range(start, end + 1):
                # Write page delimiter and number
                f.write(f"\n--- Page {pg_num} ---\n\n")
                _, text = pages[pg_num - 1]
                f.write(text.strip() + "\n")
        print(f"  Created {fname}  (pages {start}–{end})")

# ─── Logging ─────────────────────────────────────────────────────
def write_log(pages, batches, scanned_flags):
    """Create a Markdown log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_pages = len(pages)
    num_batches = len(batches)
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write("# Extraction Log\n\n")
        log.write(f"- **Total pages**: {total_pages}\n")
        log.write(f"- **Batches created**: {num_batches}\n\n")
        if scanned_flags:
            log.write("## Pages with no/low text (possible scanned images)\n\n")
            log.write("| Page | Text length |\n|------|-------------|\n")
            for pg, length in scanned_flags:
                log.write(f"| {pg} | {length} |\n")
        else:
            log.write("No pages flagged as scanned images.\n")

        log.write("\n## Batch details\n\n")
        for idx, (start, end, label) in enumerate(batches, start=1):
            log.write(f"- `batch_{idx:03d}_{label}.txt`: pages {start}–{end}\n")
    print(f"\nLog written to {LOG_PATH}")

def print_summary(pages, batches, scanned_flags):
    """Print a formatted table to console."""
    print("\n" + "="*60)
    print(f"{'Extraction Summary':^60}")
    print("="*60)
    print(f"Total pages processed : {len(pages)}")
    print(f"Batches generated     : {len(batches)}")
    if scanned_flags:
        print(f"Pages flagged (OCR)   : {len(scanned_flags)}")
    else:
        print("No pages flagged for OCR.")
    print("-"*60)
    # Table header
    print(f"{'Batch':<12} {'Pages':<12} {'Label'}")
    print("-"*60)
    for idx, (start, end, label) in enumerate(batches, start=1):
        name = f"batch_{idx:03d}"
        page_range = f"{start}-{end}"
        print(f"{name:<12} {page_range:<12} {label}")
    print("="*60)

# ─── Main ────────────────────────────────────────────────────────
def main():
    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        sys.exit(1)

    print(f"Opening {PDF_PATH} ...")
    pages = extract_pages(PDF_PATH)
    total_pages = len(pages)
    print(f"Extracted text from {total_pages} pages.")

    # Identify scanned‑like pages (very short text)
    scanned_flags = []
    for pg_num, text in pages:
        if len(text.strip()) < MIN_TEXT_LEN_FOR_VALID_PAGE:
            scanned_flags.append((pg_num, len(text.strip())))
    if scanned_flags:
        print(f"Warning: {len(scanned_flags)} pages have almost no text – possible scanned images.")

    # Detect Kanda segments
    segments = kanda_ranges(pages)
    print(f"Detected {len(segments)} Kanda segment(s).")

    # Choose splitting strategy
    batches = choose_best_batches(pages, segments)

    # Write output files
    print("\nWriting batch files ...")
    write_batches(pages, batches)

    # Write log
    write_log(pages, batches, scanned_flags)

    # Print summary
    print_summary(pages, batches, scanned_flags)

if __name__ == "__main__":
    main()