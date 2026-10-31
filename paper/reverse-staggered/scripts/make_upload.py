"""Build the Overleaf/arXiv upload folder and zip from the current sources.

Copies main.tex, refs.bib and figures/*.tex into arxiv-upload/ (replacing it),
refuses absolute Windows/Unix paths in the copied files, and writes
arxiv-upload.zip with paths relative to the folder. Run make_figures.py first.

Usage (from the repository root):
  D:\\Code\\venv313\\Scripts\\python.exe paper/reverse-staggered/scripts/make_upload.py
"""

from pathlib import Path
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "arxiv-upload"
FILES = ["main.tex", "refs.bib", *sorted(p.relative_to(HERE).as_posix() for p in (HERE / "figures").glob("*.tex"))]
# A lone drive letter (not the end of a word such as "Last:\\") or a Unix home/tmp root.
ABSOLUTE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home|tmp)/")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for name in FILES:
        text = (HERE / name).read_text(encoding="utf-8")
        if ABSOLUTE.search(text):
            raise ValueError(f"{name} contains an absolute path")
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    archive = HERE / "arxiv-upload.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.write(OUT / name, name)
    print(archive.name, FILES)


if __name__ == "__main__":
    main()
