"""report.html for a run folder: the preview video and how to load the data."""
from __future__ import annotations

import html
import json
from pathlib import Path

FILES = [
    ("<ue>/spectrum/NNNNNN.npy", "28 GHz spatial spectrum per step (dB), same view as the camera"),
    ("<ue>/rgb/NNNNNN.jpg", "camera view per step"),
    ("<ue>/beams.csv", "power of the 55 receive beams per step (dB)"),
    ("<ue>/frames.csv", "device pose, best beam and path to the base station per step"),
    ("<ue>/arrivals.csv", "strongest arrival directions per step"),
    ("<ue>/observations.csv", "the people this device sees per step"),
    ("pedestrians.csv", "where every person is"),
    ("meta.json", "scene, camera, receive beams, the scenario"),
]


def write_report(run: str | Path) -> Path:
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    ue = meta["ues"][0] if meta["ues"] else "ue0"
    bs = next(iter(meta["base_stations"]))
    snippet = f"""import numpy as np, pandas as pd
from PIL import Image

k = 10                                                        # step
spectrum = np.load(f"{run.name}/{ue}/spectrum/{{k:06d}}.npy")   # 28 GHz spectrum (dB)
camera   = Image.open(f"{run.name}/{ue}/rgb/{{k:06d}}.jpg")
beams    = pd.read_csv(f"{run.name}/{ue}/beams.csv")          # 55 receive beams per step
frames   = pd.read_csv(f"{run.name}/{ue}/frames.csv")         # pose, best beam, path to {bs}"""
    rows = "".join(f"<tr><td><code>{html.escape(a.replace('<ue>', ue))}</code></td><td>{html.escape(b)}</td></tr>" for a, b in FILES)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadioSight · {html.escape(run.name)}</title>
<style>
:root {{ --bg:#faf9f6; --surface:#fff; --ink:#18283e; --text:#34465d; --muted:#53667c; --line:#dbe3ec; --accent:#295fa8; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:28px 20px 56px; }}
.kicker {{ color:var(--accent); font-weight:800; letter-spacing:.18em; text-transform:uppercase; font-size:12px; }}
h1 {{ color:var(--ink); margin:6px 0 18px; font-size:28px; letter-spacing:-.02em; }}
h2 {{ color:var(--ink); margin:34px 0 10px; font-size:19px; }}
video {{ width:100%; border-radius:14px; background:#0b1220; display:block; }}
pre {{ background:#132033; color:#e7edf5; border-radius:12px; padding:14px 16px; overflow-x:auto; font-size:13px; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; background:var(--surface); border:1px solid var(--line); }}
td {{ padding:7px 12px; border-bottom:1px solid var(--line); }} td:first-child {{ white-space:nowrap; }}
</style></head><body><main>
<div class="kicker">RadioSight · generated data</div>
<h1>{html.escape(meta["twin"])} · {html.escape(meta.get("scenario", {}).get("_file", run.name))}</h1>
<video src="preview.mp4" controls autoplay muted loop playsinline></video>
<h2>Load it</h2>
<pre>{html.escape(snippet)}</pre>
<table>{rows}</table>
</main></body></html>"""
    out = run / "report.html"
    out.write_text(page)
    return out
