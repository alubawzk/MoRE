import mujoco
from mujoco import viewer
import os
import numpy as np
import time
from pdb import set_trace as st
current_dir = os.path.dirname(os.path.abspath(__file__))
mjcf_path   = os.path.join(current_dir, "./mjcf/mini3_box.xml")
#mjcf_path   = os.path.join(current_dir, "robot/robot.mjcf")
#st()
model = mujoco.MjModel.from_xml_path(mjcf_path)
data  = mujoco.MjData(model)

"""
## pring body names
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
              for i in range(model.nbody)]
for i, name in enumerate(body_names):
    print(f"Body {i}: {name}")

## print qvel
print("qvel (关节速度向量):")
for i, v in enumerate(data.qvel):
    print(f"qvel[{i}] = {v}")

## print joint names
print("Joint names:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"joint[{i}] = {name}")

# joint name 到 qvel      <inertial pos="0.000030 0.009550 -0.026636" mass="0.59695997" diaginertia="0.00297 0.003538 0.002153" /> 索引映射字典
joint_qvel_mapping = {}

for dof_index in range(model.nv):
    joint_id = model.dof_jntid[dof_index]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

    if joint_name is None:
        joint_name = f"unnamed_joint_{joint_id}"

    if joint_name not in joint_qvel_mapping:
        joint_qvel_mapping[joint_name] = []

    joint_qvel_mapping[joint_name].append(dof_index)
# 打印映射表
print("Joint name → qvel index mapping:")
for joint_name, indices in joint_qvel_mapping.items():
    print(f"{joint_name}: qvel indices {indices}")
"""


for i in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"actuator[{i}] = {aname}")


def add_com_visuals(scn, com):
    """在场景中绘制重心球体、垂直投影线和地面投影点"""
    ground = np.array([com[0], com[1], 0.0])
    mat = np.eye(3).flatten().astype(np.float64)

    # 红色球体标记重心
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.01, 0.01, 0.01]),
            com.copy(),
            mat,
            np.array([1.0, 0.0, 0.0, 1.0])
        )
        g.rgba[:] = [1.0, 0.0, 0.0, 1.0]  # 强制设置红色
        scn.ngeom += 1

    # # 红色竖线从重心投影到地面
    # if scn.ngeom < scn.maxgeom:
    #     g = scn.geoms[scn.ngeom]
    #     mujoco.mjv_connector(
    #         g,
    #         mujoco.mjtGeom.mjGEOM_CYLINDER,
    #         0.012,
    #         com,
    #         ground
    #     )
    #     g.rgba[:] = [1.0, 0.0, 0.0, 1.0]  # 红色竖线
    #     scn.ngeom += 1

    # 红色圆盘标记地面投影点
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([0.008, 0.008, 0.001]),
            ground.copy(),
            mat,
            np.array([1.0, 0.0, 0.0, 1.0])
        )
        g.rgba[:] = [1.0, 0.0, 0.0, 1.0]  # 红色投影点
        scn.ngeom += 1


def compute_com(model, data):
    """手动计算全身重心（跳过 world body）"""
    total_mass = 0.0
    com = np.zeros(3)
    for i in range(1, model.nbody):
        m = model.body_mass[i]
        total_mass += m
        com += m * data.xipos[i]
    if total_mass > 0:
        com /= total_mass
    return com


## 启动可视化
with viewer.launch_passive(model, data) as v:
    # 打印 user_scn 容量，方便调试
    print(f"[debug] user_scn.maxgeom = {v.user_scn.maxgeom}")
    step_count = 0
    while v.is_running():
        t = data.time
        step_count += 1
        data.ctrl[16] = 500 * np.sin(2 * np.pi * 0.5 * t)
        mujoco.mj_step(model, data)

        # 用锁保证线程安全地修改 user_scn
        with v.lock():
            v.user_scn.ngeom = 0
            com = compute_com(model, data)
            add_com_visuals(v.user_scn, com)

        v.sync()
        time.sleep(0.001)
        st()
 
