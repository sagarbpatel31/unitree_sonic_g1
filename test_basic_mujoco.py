#!/usr/bin/env python3
"""
Basic MuJoCo Test - Verify installation works
"""

import mujoco
import numpy as np

# Simple test XML
xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <geom type="box" size="0.1 0.1 0.1"/>
      <joint type="free"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    print("🧪 Testing MuJoCo installation...")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print("✅ MuJoCo model created successfully")
    print(f"   📊 Model: {model.nq} DOFs, {model.nu} actuators")

    # Test simulation
    for i in range(10):
        mujoco.mj_step(model, data)

    print(f"✅ Physics simulation works")
    print(f"   📍 Box position: {data.qpos[:3]}")

    print("\n🎯 MuJoCo is working perfectly!")
    print("   For 3D visualization on macOS, use: mjpython script.py")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()