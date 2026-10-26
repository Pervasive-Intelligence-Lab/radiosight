"""Download the twin, the recorded activities and the people from the GitHub release of this repository."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path


def _manifest() -> dict:
    m = json.loads(resources.files("radiosight_sim").joinpath("assets.json").read_text())
    m["base_url"] = os.environ.get("RADIOSIGHT_ASSETS_URL", m["base_url"])  # mirrors / local testing
    return m


def download(name: str, dest: Path) -> None:
    m = _manifest()
    names = list(m["assets"]) if name == "all" else [name]
    for n in names:
        if n not in m["assets"]:
            raise SystemExit(f"unknown asset {n!r}; available: {', '.join(m['assets'])}, all")
        a = m["assets"][n]
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / n).exists():
            print(f"{dest / n} exists, skipping")
            continue
        url = f"{m['base_url']}/{a['file']}"
        tmp = dest / (a["file"] + ".part")
        print(f"downloading {url} ({a['bytes'] / 1e6:.0f} MB)")
        h = hashlib.sha256()
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != a["sha256"]:
            tmp.unlink()
            raise SystemExit(f"checksum mismatch for {a['file']}")
        with zipfile.ZipFile(tmp) as z:
            z.extractall(dest)
        tmp.unlink()
        print(f"  -> {dest / n}")
