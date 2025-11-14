#!/usr/bin/env python3
"""
extract_signature_from_code.py (improved)

Finds Python `def` or `class` signatures inside `code` when `signature` is empty,
moves the signature string into `signature`, removes the first occurrence of it
from the `code` field, and writes results to outfile.

Usage:
  python extract_signature_from_code.py --infile data/python_data.json --outfile data/python_data_with_sig.json --show_examples
"""
import re, json, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="data/python_data.json")
parser.add_argument("--outfile", type=str, default="data/python_data_with_sig.json")
parser.add_argument("--show_examples", action="store_true")
parser.add_argument("--max_examples", type=int, default=5)
args = parser.parse_args()

# Improved pattern:
# - optional "async "
# - matches 'def' or 'class'
# - name (identifier)
# - optional parentheses (anything but a closing paren) -> handles generics, types, commas
# - optional trailing colon (kept if present)
sig_pattern = re.compile(
    r"""(?P<sig>
        (?:async\s+)?                         # optional 'async'
        (?:def|class)                         # def or class
        \s+[A-Za-z_]\w*                       # identifier
        (?:\s*\([^)]*\))?                     # optional parentheses with anything except ')'
        \s*:?                                 # optional colon (will be included if present)
    )""",
    re.VERBOSE
)

def normalize_leading_markers(s):
    # remove leading markup tokens common in this dataset and leading whitespace
    return re.sub(r'^(?:\s|(<NEW_LINE>|<NEWLINE>|<INDENT>|<DEDENT>|\n|\r\n))+', '', s)

count_total = 0
count_extracted = 0
examples = []

os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

with open(args.infile, "r", encoding="utf8") as fin, open(args.outfile, "w", encoding="utf8") as fout:
    for line in fin:
        count_total += 1
        rec = json.loads(line)
        sig = (rec.get("signature") or "").strip()
        code = rec.get("code") or ""
        if not sig and code:
            # Try to find signature near start first (faster & safer)
            m_start = sig_pattern.search(code[:300])  # common signatures appear early
            m_any = None if m_start else sig_pattern.search(code)
            m = m_start if m_start else m_any
            if m:
                extracted = m.group("sig").strip()
                # Remove only the first occurrence of the matched snippet
                new_code = code.replace(m.group("sig"), "", 1)
                # Normalize leading special markers
                new_code = normalize_leading_markers(new_code)
                rec["signature"] = extracted
                rec["code"] = new_code
                count_extracted += 1
                if len(examples) < args.max_examples:
                    examples.append((rec.get("_id", str(count_total-1)), extracted, new_code[:200].replace("\n","\\n")))
        # ensure _id is string
        rec["_id"] = str(rec.get("_id", count_total-1))
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Processed {count_total} records. Extracted signatures for {count_extracted} records.")
if args.show_examples:
    print("\nExamples (id, signature, code-prefix):")
    for ex in examples:
        print(ex)
