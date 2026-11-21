"""Inline results.json into the report template and write one self-contained page.

    python3 -m evo.build_report --results results.json --out report.html

The output is a page fragment in the form the Artifact tool expects (it supplies
the doctype, head and body). It also opens directly in a browser.
"""
import argparse
import json
import os
import sys

MARKER = "/*__RESULTS__*/null"


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--template", default=os.path.join(here, "report_template.html"))
    ap.add_argument("--out", default="report.html")
    a = ap.parse_args()

    with open(a.results) as f:
        results = json.load(f)
    with open(a.template) as f:
        template = f.read()
    if MARKER not in template:
        sys.exit(f"Template {a.template} has no {MARKER} marker.")

    # "</" must not appear raw inside a <script> block; U+2028/2029 break JS string literals.
    payload = (json.dumps(results, ensure_ascii=False, separators=(",", ":"))
               .replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    page = template.replace(MARKER, payload, 1)
    with open(a.out, "w") as f:
        f.write(page)
    kb = len(page.encode("utf-8")) / 1024
    print(f"{a.out}: {kb:.0f} KB, verdict {results.get('verdict', {}).get('status')}, "
          f"mock {results.get('mock')}")


if __name__ == "__main__":
    main()
