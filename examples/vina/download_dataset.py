"""Download and safely extract the official PoseBusters paper dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
from urllib.request import urlopen
import zipfile


URL = (
    "https://zenodo.org/records/8278563/files/"
    "posebusters_paper_data.zip?download=1"
)
EXPECTED_MD5 = "f004ac7c4e68317b5348497d2bb6bee6"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "data/raw"
ARCHIVE_NAME = "posebusters_paper_data.zip"
EXTRACTED_DIR = "posebusters_benchmark_set"
TOP_LEVEL_FILES = {"README.txt", "posebusters_benchmark_set_ids.txt"}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published checksum is MD5
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        name in TOP_LEVEL_FILES
        or (path.parts and path.parts[0] == EXTRACTED_DIR)
    )


def safe_extract_posebusters(archive: Path, destination: Path) -> None:
    """Extract only PoseBusters files and reject path traversal entries."""

    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if not _selected_member(member.filename):
                continue
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="verify and extract an archive already present in --raw-dir",
    )
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    archive = args.raw_dir / ARCHIVE_NAME
    if not args.extract_only and (args.force or not archive.exists()):
        partial = archive.with_suffix(archive.suffix + ".part")
        with urlopen(URL) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(archive)
    if not archive.exists():
        raise FileNotFoundError(
            f"archive not found for --extract-only: {archive}"
        )

    observed = md5sum(archive)
    if observed != EXPECTED_MD5:
        raise RuntimeError(
            "PoseBusters archive MD5 mismatch: "
            f"expected {EXPECTED_MD5}, got {observed}"
        )
    safe_extract_posebusters(archive, args.raw_dir)
    print(f"verified={archive} md5={observed}")
    print(f"extracted={args.raw_dir / EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
