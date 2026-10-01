#!/usr/bin/env python3
"""Download the public CLA subset of the Kaya et al. (2018) EEG dataset.

Verified public source (2026-10-01): the dataset's official deposition is
the Figshare collection linked from the paper (DOI 10.1038/sdata.2018.211,
"Data Records: ... can be accessed via the Figshare data deposition
service"):

    https://doi.org/10.6084/m9.figshare.c.3917698

It hosts 17 CLA experiment records as individual MATLAB 5.0 ``.mat``
files (~40 MB each) with direct public download URLs (ndownloader) and
MD5 checksums via the Figshare API. No login required. The Harvard
Dataverse DOI sometimes quoted for this dataset (10.7910/DVN/WD8CDM) does
NOT resolve and was discarded.

Each ``.mat`` contains the MATLAB struct ``o`` with fields ``data``
(nS x 22 EEG channels at 200 Hz) and ``marker`` (nS x 1 event codes:
1 left hand, 2 right hand, 3 passive, 0 blank, 99 relaxation start,
91 breaks, 92 end) — exactly the format consumed by
:func:`eeg_classifier.data.load_file`.

Usage::

    python scripts/download_data.py                 # all 17 CLA files to data/raw
    python scripts/download_data.py --limit 1       # smoke test
    python scripts/download_data.py --paradigm all  # every paradigm (~13 GB)

After downloading, preprocess sessions into voxel batches with
``eeg_classifier.data.prepare_subject``.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

FIGSHARE_API = "https://api.figshare.com/v2"
COLLECTION_DOI = "10.6084/m9.figshare.c.3917698"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

CHUNK_SIZE = 1024 * 1024
PROGRESS_EVERY_BYTES = 10 * CHUNK_SIZE


def api_get_json(url: str):
    """GET a JSON document from the Figshare API."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_collection_id(collection_doi: str) -> int:
    """Resolve a Figshare collection DOI to its numeric id."""
    url = f"{FIGSHARE_API}/collections/:persistentId?persistentId=doi:{collection_doi}"
    try:
        return api_get_json(url)["id"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"could not resolve collection DOI {collection_doi} via {url}: {exc}\n"
            "check your network connection; the collection must exist at "
            f"https://doi.org/{collection_doi}")


def list_experiment_articles(collection_id: int, paradigm: str) -> list[dict]:
    """List experiment records of the collection, optionally one paradigm."""
    articles = api_get_json(
        f"{FIGSHARE_API}/collections/{collection_id}/articles?page_size=100")
    experiments = [a for a in articles if a.get("title", "").startswith("Experiment")]
    if paradigm != "all":
        wanted = paradigm.upper()
        experiments = [a for a in experiments if wanted in a.get("title", "").upper()]
    return sorted(experiments, key=lambda a: a["title"])


def md5_of_file(path: Path) -> str:
    """Streaming MD5 of a local file."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, expected_size: int, expected_md5: str | None) -> None:
    """Download ``url`` to ``target`` with progress output and MD5 check.

    Skips the download when a complete file of the expected size already
    exists; incomplete files are restarted from scratch.
    """
    if target.exists() and target.stat().st_size == expected_size:
        print(f"  already complete, skipping: {target.name}")
    else:
        print(f"  downloading {target.name} ({expected_size / (1024 * 1024):.1f} MB)")

        request = urllib.request.Request(url, headers={"User-Agent": "eeg-classifier/0.1"})

        downloaded = 0
        next_report = PROGRESS_EVERY_BYTES

        with urllib.request.urlopen(request) as response, open(target, "wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"    {100.0 * downloaded / expected_size:>2.0f}% ({downloaded / (1024 * 1024):.1f} MB)")
                    next_report += PROGRESS_EVERY_BYTES

    if expected_md5:
        actual = md5_of_file(target)
        if actual != expected_md5:
            raise SystemExit(
                f"MD5 mismatch for {target}: expected {expected_md5}, got {actual}. Delete the file and retry."
                )
        print("  md5 ok")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download the public Kaya et al. (2018) EEG dataset from Figshare.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="target directory (default data/raw).")
    parser.add_argument("--paradigm", default="CLA",
                        help="'CLA' (default), or any of CLA/5F/HALT/FREEFORM/NOMT, or 'all'.")
    parser.add_argument("--limit", type=int, default=None,
                        help="download at most N experiment records (for testing).")
    parser.add_argument("--collection", default=COLLECTION_DOI,
                        help="override the Figshare collection DOI.")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip MD5 verification after download.")
    args = parser.parse_args(argv)

    collection_id = resolve_collection_id(args.collection)
    articles = list_experiment_articles(collection_id, args.paradigm)

    if args.limit is not None:
        articles = articles[:args.limit]

    if not articles:
        raise SystemExit(f"no experiment records found for paradigm '{args.paradigm}'")

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"collection : https://doi.org/{args.collection}")
    print(f"paradigm   : {args.paradigm}")
    print(f"records    : {len(articles)}")
    print(f"target dir : {args.out}")
    print()

    failures = 0

    for article in articles:
        print(article["title"])

        details = api_get_json("{}/articles/{}".format(FIGSHARE_API, article["id"]))

        for file_info in details.get("files", []):
            target = args.out / file_info["name"]
            try:
                download_file(file_info["download_url"], target,
                              file_info["size"],
                              None if args.no_verify else file_info.get("computed_md5"))
            except SystemExit:
                raise
            except (OSError, KeyError) as exc:
                print(f"  FAILED: {exc}")
                failures += 1

        print()

    if failures:
        raise SystemExit(f"{failures} file(s) failed; re-run the script to retry.")

    print("done. Next step: preprocess a session into voxel batches, e.g.")
    print("  PYTHONPATH=src python -c \"from eeg_classifier.data import prepare_subject; "
          "prepare_subject('data/raw/CLA-SubjectA-160108-3St-LRHand.mat', 0)\"")


if __name__ == "__main__":
    main()
