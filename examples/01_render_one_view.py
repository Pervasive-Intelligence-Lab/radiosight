"""Render one camera view and its 28 GHz spectrum from a twin.

python examples/01_render_one_view.py twins/nlos_tx1
"""
import sys

import numpy as np
from PIL import Image

from radiosight_sim import Pose, RadioTwin
from radiosight_sim.viz import _turbo

twin = RadioTwin.load(sys.argv[1] if len(sys.argv) > 1 else "twins/nlos_tx1")
print(twin)

# stand in the middle of the recorded walk and look at the base station
pose = Pose.look_at(twin.landmark("ue_area"), twin.landmark("bs1"))
r = twin.render(pose, width=640)
lo, hi = twin.window_db
Image.fromarray(np.concatenate([r.rgb, _turbo((r.rf_db - lo) / (hi - lo))], 1)).save("view.png")
v, u = np.unravel_index(np.argmax(r.rf_db), r.rf_db.shape)
print(f"strongest direction at pixel ({u}, {v}): {r.rf_db[v, u]:.1f} dB; wrote view.png")
