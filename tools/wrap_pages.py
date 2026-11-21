"""Turn the page fragments the build scripts write into standalone pages under docs/.

    python3 tools/wrap_pages.py

The build scripts (evo.build_report, evo2.build_report, evo3.report3, evo3.breadth_page,
summary_build.py) write a page body with its own <title> and <style> but no document head.
This adds the doctype, character set and viewport so the page opens correctly in a browser
or on GitHub Pages. docs/index.html is a copy of the summary page.
"""
import os

PAGES = ["summary.html", "social.html", "breadth.html", "report3.html", "report2.html", "report.html"]

HEAD = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
:root { color-scheme: light }
body { margin: 0; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif }
img { max-width: 100% }
[hidden] { display: none !important }
</style>
"""


def main():
    os.makedirs("docs", exist_ok=True)
    for name in PAGES:
        if not os.path.exists(name):
            print(f"skipped {name}: build it first")
            continue
        page = HEAD + open(name, encoding="utf-8").read()
        open(os.path.join("docs", name), "w", encoding="utf-8").write(page)
        if name == "summary.html":
            open(os.path.join("docs", "index.html"), "w", encoding="utf-8").write(page)
        print(f"wrote docs/{name}")


if __name__ == "__main__":
    main()
