#!/usr/bin/env python3
"""
Test MuJoCo viewer with mjpython
"""

import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" size="2 2 0.1" type="box" rgba="0.5 0.5 0.5 1"/>
    <body name="box" pos="0 0 1">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
      <joint type="free"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    print("🧪 Testing MuJoCo with 3D viewer...")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print("✅ Model loaded, opening 3D viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✅ 3D viewer opened! Close window to exit.")

        step = 0
        while viewer.is_running() and step < 1000:
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

            if step % 100 == 0:
                print(f"   Step {step}: Box height = {data.qpos[2]:.3f}")

    print("✅ 3D viewer test completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()