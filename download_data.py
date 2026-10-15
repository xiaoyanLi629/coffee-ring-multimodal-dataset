#!/usr/bin/env python3
"""Download the dataset from figshare into data/, verify each archive and unpack it.

    python download_data.py                         # all five archives (about 480 MB)
    python download_data.py features_metadata sem   # only some archives

Uses only the Python standard library. Archives already downloaded with the
right MD5 are not fetched again.
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTICLE = 34516902                                   # doi:10.6084/m9.figshare.34516902
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "_archives"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, size):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r  {dest.name}: {done / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main(wanted):
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{ARTICLE}", timeout=60) as r:
        files = json.load(r)["files"]
    if wanted:
        files = [f for f in files if f["name"].removesuffix(".zip") in wanted]
        missing = set(wanted) - {f["name"].removesuffix(".zip") for f in files}
        if missing:
            sys.exit(f"unknown archive(s): {', '.join(sorted(missing))}")
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = CACHE / f["name"]
        if not (dest.exists() and md5(dest) == f["computed_md5"]):
            fetch(f["download_url"], dest, f["size"])
            if md5(dest) != f["computed_md5"]:
                sys.exit(f"MD5 mismatch for {dest.name}; delete it and run again")
        with zipfile.ZipFile(dest) as z:
            z.extractall(DATA)
            print(f"  {dest.name}: verified, {len(z.namelist())} entries unpacked into data/")
    print("done; the archives are kept in data/_archives/ and can be deleted")


if __name__ == "__main__":
    main(sys.argv[1:])
