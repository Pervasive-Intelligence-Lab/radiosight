"""Command line: radiosight-sim {download, info, render, run}."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def cmd_download(a):
    from .download import download
    for name in a.names:
        download(name, Path(a.dest))


def cmd_info(a):
    from .twin import RadioTwin
    tw = RadioTwin.load(a.twin)
    print(tw)
    m = tw.meta
    print(json.dumps({k: m[k] for k in ("title", "provenance", "radio") if k in m}, indent=1))
    print("trajectories:", ", ".join(m.get("trajectories", {})))
    for name in m.get("trajectories", {}):
        print(f"  {name}: {tw.trajectory(name)}")


def cmd_render(a):
    from PIL import Image
    from .geometry import Pose
    from .twin import RadioTwin
    from .viz import spectrum_rgb
    tw = RadioTwin.load(a.twin)
    eye = tw.landmark(a.at) if not a.at[0].isdigit() and not a.at.startswith("-") else np.array([float(v) for v in a.at.split(",")])
    r = tw.render(Pose.look_at(eye, tw.landmark(a.look_at)), width=a.width)
    lo, hi = tw.window_db
    out = Path(a.out)
    Image.fromarray(np.concatenate([r.rgb, spectrum_rgb(r.rf_db, lo)], 1)).save(out)
    print(f"wrote {out}: camera view | spectrum ({r.rf_db.min():.1f} .. {r.rf_db.max():.1f} dB)")


def cmd_run(a):
    from .recorder import Recorder
    from .scenario import load_scenario
    sim, cfg = load_scenario(a.scenario, twin_dir=a.twin)
    cfg["_file"] = Path(a.scenario).name
    duration = a.duration or cfg.get("duration", 30.0)
    rec_cfg = cfg.get("record", {})
    out = Path(a.out)
    rec = Recorder(out, sim, rec_cfg.get("rgb", True), rec_cfg.get("rf", True), rec_cfg.get("depth", False), scenario=cfg)
    writer = None
    if rec_cfg.get("video", True) and not a.no_video:
        import imageio.v2 as imageio
        from .viz import Previewer
        prev = Previewer(sim.twin, panel=360)
        writer = imageio.get_writer(out / "preview.mp4", fps=max(1, round(1 / sim.dt)), codec="libx264", quality=7,
                                    macro_block_size=8)
    t0, n = time.time(), 0
    for step in sim.run(duration):
        rec.write(step)
        if writer is not None:
            writer.append_data(prev.frame(step, a.preview_ue))
        n += 1
        if n % 50 == 0:
            print(f"  t = {step.t:6.1f} s   ({n / (time.time() - t0):.1f} steps/s)")
    rec.close()
    if writer is not None:
        writer.close()
    from .report import write_report
    write_report(out)
    print(f"wrote {n} steps to {out} in {time.time() - t0:.0f} s (open {out / 'report.html'})")


def cmd_report(a):
    from .report import write_report
    print(f"wrote {write_report(a.run)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="radiosight-sim", description="Generate 28 GHz data from RadioSight radio twins.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the twin, the recorded activities and the people")
    d.add_argument("names", nargs="+", help="nlos_tx1, motions, people, or all")
    d.add_argument("--dest", default="twins")
    d.set_defaults(fn=cmd_download)
    i = sub.add_parser("info", help="describe a twin")
    i.add_argument("twin")
    i.set_defaults(fn=cmd_info)
    r = sub.add_parser("render", help="render one view: camera image | spectrum")
    r.add_argument("twin")
    r.add_argument("--at", default="ue_area", help="landmark name or x,y,z")
    r.add_argument("--look-at", default="bs1")
    r.add_argument("--width", type=int, default=640)
    r.add_argument("--out", default="view.png")
    r.set_defaults(fn=cmd_render)
    s = sub.add_parser("run", help="run a scenario file and record the generated data")
    s.add_argument("scenario")
    s.add_argument("--out", required=True)
    s.add_argument("--twin", help="override the twin folder named in the scenario")
    s.add_argument("--duration", type=float)
    s.add_argument("--no-video", action="store_true")
    s.add_argument("--preview-ue", help="which UE the preview video follows (default: the first)")
    s.set_defaults(fn=cmd_run)
    rp = sub.add_parser("report", help="write report.html for a run folder (what is in it, what it looks like)")
    rp.add_argument("run")
    rp.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
