"""
download_real_datasets.py
=========================
Download real conjunctiva images from public sources.

Configured sources (call --source NAME to select one):
  - kaggle: EYES-DEFY-ANEMIA (or any Kaggle conjunctiva anemia dataset)
  - roboflow: stub for user-provided export URL

Outputs:
  datasets/anemia_real/raw/<source_name>/<files...>
  datasets/anemia_real/raw/MANIFEST.json (per-source provenance)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("maternin.training.download")


@dataclass
class ManifestEntry:
    source: str
    url: str
    count: int
    license: str
    downloaded_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_manifest_entry(source: str, url: str, count: int, license: str) -> dict:
    """Construct a single manifest entry with required provenance fields."""
    entry = ManifestEntry(
        source=source,
        url=url,
        count=count,
        license=license,
        downloaded_at=datetime.datetime.utcnow().isoformat() + "Z",
    )
    return entry.to_dict()


def save_manifest(entries: list[dict], path: str) -> None:
    """Write JSON manifest of downloaded datasets."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2))


def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """Call fn() with exponential backoff on exception."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {exc}. Retrying in {delay}s")
            time.sleep(delay)
    raise last_exc


def download_kaggle(source_slug: str, output_dir: str) -> int:
    """Download a Kaggle dataset by slug. Returns file count."""
    target = Path(output_dir) / source_slug
    target.mkdir(parents=True, exist_ok=True)

    def _kaggle_pull():
        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = env.get("KAGGLE_USERNAME", "")
        env["KAGGLE_KEY"] = env.get("KAGGLE_KEY", "")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", source_slug,
             "-p", str(target), "--unzip"],
            check=True, env=env,
        )

    retry_with_backoff(_kaggle_pull, max_attempts=3)
    return sum(1 for _ in target.rglob("*") if _.is_file())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/anemia_real/raw")
    parser.add_argument("--kaggle-slug", default="harshwardhanfartale/eyes-defy-anemia")
    parser.add_argument("--source", choices=["kaggle"], default="kaggle")
    args = parser.parse_args()

    entries: list[dict] = []
    if args.source == "kaggle":
        try:
            count = download_kaggle(args.kaggle_slug, args.output_dir)
            entries.append(
                build_manifest_entry(
                    source=args.kaggle_slug,
                    url=f"https://www.kaggle.com/datasets/{args.kaggle_slug}",
                    count=count,
                    license="Kaggle Terms",
                )
            )
        except Exception as exc:
            logger.error(f"Kaggle download failed: {exc}")
            raise SystemExit(1)

    save_manifest(entries, os.path.join(args.output_dir, "MANIFEST.json"))
    print(f"Manifest: {len(entries)} source(s) downloaded to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
