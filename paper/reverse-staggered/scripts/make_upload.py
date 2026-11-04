"""Build the arXiv upload folder and zip from the current sources.

Copies main.tex, its BibTeX output build/main.bbl (arXiv does not run BibTeX), the ICLR
style file, refs.bib and figures/*.tex into arxiv-upload/ (replacing it). It refuses absolute
Windows/Unix paths in the copied files and a .bbl older than main.tex or refs.bib, and writes
arxiv-upload.zip with paths relative to the folder.

Compile first, keeping the intermediate files (from paper/reverse-staggered, in PowerShell):
  $env:TECTONIC_CACHE_DIR = 'E:\\APP\\tectonic\\cache'
  & 'E:\\APP\\tectonic\\tectonic.exe' -X compile --outdir build --keep-intermediates main.tex

Usage (from the repository root):
  D:\\Code\\venv313\\Scripts\\python.exe paper/reverse-staggered/scripts/make_upload.py
"""

from pathlib import Path
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "arxiv-upload"
SOURCE, BBL, STYLE = "main.tex", "build/main.bbl", "iclr2027_conference.sty"
# (source path, path inside the upload)
FILES = [(SOURCE, "main.tex"), (BBL, "main.bbl"), (STYLE, STYLE), ("refs.bib", "refs.bib"),
         *((p, p) for p in sorted(p.relative_to(HERE).as_posix() for p in (HERE / "figures").glob("*.tex")))]
# A lone drive letter (not the end of a word such as "Last:\\") or a Unix home/tmp root.
ABSOLUTE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home|tmp)/")


def main():
    bbl = HERE / BBL
    if not bbl.exists():
        raise FileNotFoundError(f"{BBL} is missing: compile with --keep-intermediates first")
    stale = [n for n in (SOURCE, "refs.bib") if (HERE / n).stat().st_mtime > bbl.stat().st_mtime]
    if stale:
        raise RuntimeError(f"{BBL} is older than {stale}: recompile with --keep-intermediates")
    if OUT.exists():
        shutil.rmtree(OUT)
    for name, dest in FILES:
        (OUT / dest).parent.mkdir(parents=True, exist_ok=True)
        if name == STYLE:  # third-party template file, copied unchanged
            shutil.copyfile(HERE / name, OUT / dest)
            continue
        text = (HERE / name).read_text(encoding="utf-8")
        if ABSOLUTE.search(text):
            raise ValueError(f"{name} contains an absolute path")
        (OUT / dest).write_text(text, encoding="utf-8", newline="\n")
    archive = HERE / "arxiv-upload.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, dest in FILES:
            zf.write(OUT / dest, dest)
    print(archive.name, [dest for _, dest in FILES])


if __name__ == "__main__":
    main()
