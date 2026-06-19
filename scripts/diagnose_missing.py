import re
from pathlib import Path

text = Path("cleaned_text/full_corpus.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
lines = text.split("\n")

# every raw "NN సర్గలు" occurrence, regardless of whether it became a confirmed anchor
raw_hits = [i for i, l in enumerate(lines) if re.search(r"[\d,\s]+సర్గలు\s*$", l.strip())]
print("raw సర్గలు line count:", len(raw_hits))

# now re-run your actual anchor-finding logic to see which ones got confirmed
import importlib.util
spec = importlib.util.spec_from_file_location("parser", "scripts/02_parse_final.py")
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

anchors = parser.find_chapter_anchors(lines)
confirmed_lines = {a["start_line"] for a in anchors}

missing = [i for i in raw_hits if i not in confirmed_lines]
print("unconfirmed సర్గలు lines:", len(missing))
for i in missing:
    print("\n--- line", i, "---")
    for j in range(max(0, i-2), min(len(lines), i+6)):
        print(j, repr(lines[j]))