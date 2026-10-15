"""Download the large prediction tables for this software release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(archive: Path, specification: dict, manifest: dict) -> None:
    if archive.stat().st_size != specification["bytes"]:
        raise ValueError("Archive size differs from the release manifest")
    if checksum(archive) != specification["sha256"]:
        raise ValueError("Archive checksum differs from the release manifest")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    with zipfile.ZipFile(archive) as bundle:
        for name in specification["files"]:
            expected = entries[name]
            destination = (ROOT / name).resolve()
            if not destination.is_relative_to(ROOT):
                raise ValueError("Data path is outside the repository")
            if destination.exists():
                if destination.stat().st_size == expected["bytes"] and checksum(destination) == expected["sha256"]:
                    print(f"Already present: {name}")
                    continue
                raise FileExistsError(f"Existing file differs; left unchanged: {name}")
            info = bundle.getinfo(name)
            if info.file_size != expected["bytes"]:
                raise ValueError(f"Unexpected uncompressed size: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    with bundle.open(info) as source:
                        shutil.copyfileobj(source, temporary)
                except BaseException:
                    temporary.close()
                    temporary_path.unlink(missing_ok=True)
                    raise
            try:
                if checksum(temporary_path) != expected["sha256"]:
                    raise ValueError(f"Prediction-table checksum differs: {name}")
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            print(f"Restored: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded release archive")
    arguments = parser.parse_args()
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    specification = manifest["data_archive"]
    if arguments.archive:
        restore(arguments.archive.resolve(), specification, manifest)
        return
    with tempfile.TemporaryDirectory(prefix="nv-release-") as directory:
        archive = Path(directory) / "predictions.zip"
        request = urllib.request.Request(specification["url"], headers={"User-Agent": "nv-magnetic-localisation/1.0.0"})
        print(f"Downloading {specification['url']}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        restore(archive, specification, manifest)


if __name__ == "__main__":
    main()
