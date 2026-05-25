# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process OakInk-v2 raw data: 4-second clip starting 2s before grasp.

This pipeline reads directly from the original OakInk-v2 ``anno_preview``
pickle (downloaded from ``kelvin34501/OakInk-v2``) plus the ``program``
primitive segmentation. It crops the trajectory to a 4-second window that
starts 2 seconds before the primitive's grasp moment, so the resulting clip
includes the approach phase where the hand is *not* already on the object.

Compared to ``oakink.py`` this pipeline:

1. Uses raw OakInk-v2 (not maniptrans-preprocessed pickles).
2. Crops to 2s pre-grasp + 2s during-grasp = 4s total at 50Hz.
3. Operates on a single hand (right) and zero-fills the left side so the
   downstream NPZ schema (``decompose_fast``/``ik_fast``) still matches.
4. Visualizes via :mod:`spider.viewers.viser_viewer` with a 21-joint MANO
   skeleton (when ``--mano-assets-root`` is provided) or fingertip-only
   markers as a fallback.

Author: Chaoyi Pan
Date: 2026-05-24
"""

import ast
import io
import json
import os
import pickle
import time
from pathlib import Path

import loguru
import mujoco
import numpy as np
import pymeshlab
import torch
import tyro
from scipy.spatial.transform import Rotation as R

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# OakInk-v2 mocap rate
FPS_MOCAP = 120
# manotorch MANO 21-joint indices for the 5 fingertips (SNAP layout)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
# bone connectivity for the 21-joint skeleton (parent -> child)
SKELETON_BONES = [
    # thumb: wrist->TMC->TMP->TDP->TIP
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index: wrist->MCP->PIP->DIP->TIP
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def _seq_token_from_seq_key(seq_key: str) -> str:
    return seq_key.replace("/", "++")


def _try_load_mano_layer(mano_assets_root: str | None):
    """Return a manotorch ManoLayer for right hand, or None if unavailable."""
    if mano_assets_root is None:
        return None
    mano_assets_root = os.path.expanduser(mano_assets_root)
    if not os.path.exists(os.path.join(mano_assets_root, "models", "MANO_RIGHT.pkl")):
        loguru.logger.warning(
            f"MANO_RIGHT.pkl not found under {mano_assets_root}/models/. "
            "Skeleton will fall back to fingertip-only display."
        )
        return None
    try:
        from manotorch.manolayer import ManoLayer
    except ImportError:
        loguru.logger.warning(
            "manotorch is not installed; skeleton will fall back to "
            "fingertip-only display. Install with `uv pip install --no-deps "
            "git+https://github.com/lixiny/manotorch.git`."
        )
        return None
    return ManoLayer(
        mano_assets_root=mano_assets_root,
        rot_mode="quat",
        side="right",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    )


def _find_grasp_primitive_for_obj(
    program_info: dict, obj_id: str, hand: str = "rh"
) -> tuple[str, tuple[int, int]] | None:
    """Find the earliest primitive whose start marks ``hand``'s grasp of ``obj_id``.

    Returns (primitive_name, (start_mocap_fid, end_mocap_fid)). We pick the
    chronologically first primitive whose per-hand obj_list contains obj_id
    and whose per-hand frame_range is non-None: that primitive's start frame
    is when the hand has just acquired the object, so backing up by N seconds
    from there captures the approach phase.
    """
    obj_key = f"obj_list_{hand}"
    candidates = []
    for k, v in program_info.items():
        try:
            lh_iv, rh_iv = ast.literal_eval(k)
        except Exception:
            continue
        iv = rh_iv if hand == "rh" else lh_iv
        if iv is None:
            continue
        objs = v.get(obj_key) or []
        if obj_id not in objs:
            continue
        prim = v.get(f"primitive_{hand}") or v.get("primitive", "")
        candidates.append((iv[0], prim, iv))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    start, prim, iv = candidates[0]
    return prim, (iv[0], iv[1])


def _compute_mano_joints_world(
    mano_layer,
    pose_coeffs: torch.Tensor,
    betas: torch.Tensor,
    tsl: torch.Tensor,
) -> np.ndarray:
    """Run MANO FK and add wrist translation. Returns (T, 21, 3) numpy."""
    with torch.no_grad():
        out = mano_layer(pose_coeffs=pose_coeffs, betas=betas)
        joints = out.joints + tsl[:, None, :]
    return joints.cpu().numpy()


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    oakink2_prefix: str = f"{spider.ROOT}/../example_datasets/raw/oakinkv2",
    embodiment_type: str = "right",
    task: str = "pick_spoon_bowl",
    pre_grasp_seconds: float = 1.5,
    post_grasp_seconds: float = 2.5,
    target_fps: float = 50.0,
    mano_assets_root: str | None = None,
    show_viewer: bool = True,
):
    """Process a raw OakInk-v2 sequence: pre_grasp_seconds before + post_grasp_seconds after grasp onset."""
    if embodiment_type != "right":
        raise NotImplementedError(
            "oakinkv2 currently supports only embodiment_type='right'."
        )

    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    oakink2_prefix = os.path.abspath(os.path.expanduser(oakink2_prefix))
    if mano_assets_root is not None:
        mano_assets_root = os.path.abspath(os.path.expanduser(mano_assets_root))

    # 1. resolve task -> seq_key via the maniptrans pickle (which preserves data_path).
    maniptrans_path = f"{dataset_dir}/raw/oakink/{task}_bimanual.pkl"
    if not os.path.exists(maniptrans_path):
        raise FileNotFoundError(
            f"Maniptrans reference pickle not found: {maniptrans_path}. Available tasks: "
            f"{sorted(p.stem.replace('_bimanual','') for p in Path(f'{dataset_dir}/raw/oakink').glob('*_bimanual.pkl'))}"
        )
    with open(maniptrans_path, "rb") as f:
        mt = CPUUnpickler(f).load()
    data_path = mt["right"]["data_path"][0]
    seq_token = Path(data_path).stem
    seq_key = seq_token.replace("++", "/")

    # 2. load raw anno_preview pickle for that sequence
    anno_path = f"{oakink2_prefix}/anno_preview/{seq_token}.pkl"
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"OakInk-v2 anno not found: {anno_path}. Download via:\n"
            f"  curl -L -o {anno_path} "
            f"https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/main/anno_preview/{seq_token}.pkl"
        )
    with open(anno_path, "rb") as f:
        anno = pickle.load(f)

    # 3. determine the right-hand task target object from the maniptrans pickle.
    # This is robust: maniptrans stores the right hand's actual manipulation
    # target in mt['right']['obj_id'], whereas program_info's first rh primitive
    # may be a transient pre-grasp segment with the wrong (stationary) object.
    rh_obj_id = mt["right"]["obj_id"][0]

    # 4. load program_info and find the first non-transient primitive where the
    # right hand actively manipulates rh_obj_id.
    pi_path = f"{oakink2_prefix}/program/program_info/{seq_token}.json"
    if not os.path.exists(pi_path):
        raise FileNotFoundError(
            f"OakInk-v2 program_info not found: {pi_path}. Extract program.tar from OakInk-v2 HF dataset."
        )
    with open(pi_path) as f:
        program_info = json.load(f)
    found = _find_grasp_primitive_for_obj(program_info, rh_obj_id, hand="rh")
    if found is None:
        raise RuntimeError(
            f"No right-hand primitive found that manipulates obj_id={rh_obj_id} in {pi_path}"
        )
    primitive_name, (grasp_start, grasp_end) = found
    loguru.logger.info(
        f"task={task} seq_key={seq_key} obj={rh_obj_id} "
        f"primitive={primitive_name} grasp range=[{grasp_start}, {grasp_end}]"
    )

    # 4. compute mocap window: pre_grasp_seconds before + post_grasp_seconds after grasp_start
    pre_n = int(pre_grasp_seconds * FPS_MOCAP)
    post_n = int(post_grasp_seconds * FPS_MOCAP)
    win_start = max(0, grasp_start - pre_n)
    win_end = min(grasp_end, grasp_start + post_n)
    win_n = win_end - win_start
    total_seconds = pre_grasp_seconds + post_grasp_seconds
    loguru.logger.info(
        f"window: mocap [{win_start}, {win_end}] ({win_n} frames @ {FPS_MOCAP}Hz, "
        f"{pre_grasp_seconds}s pre-grasp + {post_grasp_seconds}s post-grasp)"
    )

    # 5. obj_id is rh_obj_id (the right hand's manipulation target from mt)
    obj_id = rh_obj_id
    if obj_id not in anno["obj_transf"]:
        raise RuntimeError(
            f"obj_id {obj_id} not found in anno['obj_transf']. "
            f"Available: {list(anno['obj_transf'].keys())}"
        )
    loguru.logger.info(f"using obj_id={obj_id}")

    # 6. extract per-frame MANO + obj data over the window
    rh_pose_coeffs = []
    rh_betas = []
    rh_tsl = []
    obj_transf = anno["obj_transf"][obj_id]
    obj_traj = []
    available = []  # which mocap fids are actually annotated in this window
    for fid in range(win_start, win_end):
        if fid not in anno["raw_mano"] or fid not in obj_transf:
            continue
        m = anno["raw_mano"][fid]
        rh_pose_coeffs.append(m["rh__pose_coeffs"])
        rh_betas.append(m["rh__betas"])
        rh_tsl.append(m["rh__tsl"])
        obj_traj.append(obj_transf[fid])
        available.append(fid)
    if not available:
        raise RuntimeError(
            f"No annotated frames in window [{win_start}, {win_end}]"
        )
    rh_pose_coeffs = torch.cat(rh_pose_coeffs, dim=0)  # (T, 16, 4)
    rh_betas = torch.cat(rh_betas, dim=0)              # (T, 10)
    rh_tsl = torch.cat(rh_tsl, dim=0)                  # (T, 3)
    obj_traj = np.stack(obj_traj, axis=0)              # (T, 4, 4)

    # 7. downsample 120Hz -> target_fps via stride
    n_out = int(total_seconds * target_fps)
    n_avail = rh_pose_coeffs.shape[0]
    stride_idx = np.linspace(0, n_avail - 1, num=min(n_out, n_avail)).round().astype(int)
    rh_pose_coeffs = rh_pose_coeffs[stride_idx]
    rh_betas = rh_betas[stride_idx]
    rh_tsl = rh_tsl[stride_idx]
    obj_traj = obj_traj[stride_idx]
    n = len(stride_idx)
    loguru.logger.info(f"downsampled to {n} frames @ {target_fps}Hz")

    # 8. compute hand keypoints (21 joints if MANO available, 5 fingertips otherwise)
    mano_layer = _try_load_mano_layer(mano_assets_root)
    if mano_layer is None:
        # fingertip-only fallback: estimate fingertips from mt pickle, time-aligned
        loguru.logger.warning(
            "MANO assets not provided. Falling back to fingertip-only display "
            "(no hand mesh, no pre-grasp finger positions). "
            "Pass --mano-assets-root=/path/to/mano_v1_2 to enable the full "
            "21-joint skeleton."
        )
        # maniptrans pickle starts at OakInk2 mocap frame grasp_start+1.
        mt_tips = mt["right"]["mano_joints"].numpy()        # (Nmt, 5, 3)
        mt_wrist = mt["right"]["wrist_pos"].numpy()         # (Nmt, 3)
        mt_first_mocap = grasp_start + 1
        # Map each output frame to its mocap fid via stride_idx.
        out_fids = [available[k] for k in stride_idx]
        wrist_traj = np.zeros((n, 3))
        tips_traj = np.zeros((n, 5, 3))
        for k, fid in enumerate(out_fids):
            if fid < mt_first_mocap:
                # pre-grasp: use raw OakInk2 wrist position; rigid-attach tips
                w = anno["raw_mano"][fid]["rh__tsl"][0].numpy()
                tips = mt_tips[0] + (w - mt_wrist[0])
            else:
                mt_idx = min((fid - mt_first_mocap) // 2, len(mt_wrist) - 1)
                w = mt_wrist[mt_idx]
                tips = mt_tips[mt_idx]
            wrist_traj[k] = w
            tips_traj[k] = tips
        wrist_world = wrist_traj
        tips_world = tips_traj
        joints_world_full = None
        full_skeleton_available = False
        wrist_rot_axisangle = np.zeros((n, 3))
    else:
        loguru.logger.info("Running MANO FK for full 21-joint skeleton.")
        joints_world_full = _compute_mano_joints_world(mano_layer, rh_pose_coeffs, rh_betas, rh_tsl)
        wrist_world = joints_world_full[:, 0, :]            # (n, 3)
        tips_world = joints_world_full[:, FINGERTIP_INDICES, :]  # (n, 5, 3)
        # wrist rotation from root quat (w,x,y,z) -> axis-angle
        root_q = rh_pose_coeffs[:, 0, :].cpu().numpy()      # (n, 4) wxyz
        scipy_q = np.concatenate([root_q[:, 1:], root_q[:, :1]], axis=1)  # xyzw
        wrist_rot_axisangle = R.from_quat(scipy_q).as_rotvec()
        full_skeleton_available = True

    # 9. set up output dirs
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="oakinkv2",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=0,
    )
    os.makedirs(output_dir, exist_ok=True)

    # 10. task_info & object mesh export
    task_info = {
        "task": task,
        "dataset_name": "oakinkv2",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": 0,
        "right_object_mesh_dir": None,
        "left_object_mesh_dir": None,
        "ref_dt": 1.0 / target_fps,
        "pre_grasp_seconds": pre_grasp_seconds,
        "post_grasp_seconds": post_grasp_seconds,
        "n_frames": n,
        # OakInk2 provenance
        "oakink_seq_key": seq_key,
        "oakink_seq_token": seq_token,
        "oakink_obj_id": obj_id,
        "oakink_primitive": primitive_name,
        "oakink_grasp_range": [grasp_start, grasp_end],
        "oakink_window": [win_start, win_end],
    }

    # Mesh: prefer the maniptrans pickle's mesh path (already resolves to obj/ply on disk).
    mt_mesh_path = mt["right"]["obj_mesh_path"][0]
    mt_mesh_name = mt_mesh_path.split("align_ds/")[1]
    mt_mesh_path = f"{dataset_dir}/raw/oakink/meshes/{mt_mesh_name}"
    if not os.path.exists(mt_mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mt_mesh_path}")
    mt_mesh_id = mt_mesh_name.split("/")[0]
    safe_name = mt_mesh_id.replace("@", "_")
    mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir, dataset_name="oakinkv2", object_name=safe_name,
    )
    os.makedirs(mesh_dir, exist_ok=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mt_mesh_path)
    ms.save_current_mesh(f"{mesh_dir}/visual.obj")
    task_info["right_object_mesh_dir"] = str(Path(mesh_dir).relative_to(dataset_dir))

    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    # 11. global rotation: oakink y-up -> mujoco z-up
    r_global = R.from_euler("xyz", [np.pi / 2, 0, 0])
    # Wrist offset for raw OakInk2 pose_coeffs (vs the maniptrans-style offset in
    # ``oakink.py``). Empirically the ManipTrans wrist axis-angle equals
    # raw_R * C with C ~ Rx(180°) * Rz(96°) (averaged across 3 sequences).
    # Composing C with oakink.py's offset (Rx(pi/2) * Rz(pi)) gives the
    # equivalent transform we should apply when starting from raw pose_coeffs.
    _C_quat_xyzw = np.array([0.67038494, 0.74167282, -0.01732714, 0.0143258])
    _C = R.from_quat(_C_quat_xyzw)
    _old_offset = R.from_euler("xyz", [np.pi / 2, 0, 0]) * R.from_euler(
        "xyz", [0, 0, np.pi]
    )
    r_right_wrist_offset = _C * _old_offset

    # 12. build qpos arrays (5 fingertips kept for downstream IK schema)
    unit_quat = np.array([1, 0, 0, 0])
    qpos_wrist_right = np.zeros((n, 7))
    qpos_finger_right = np.zeros((n, 5, 7))
    qpos_obj_right = np.zeros((n, 7))
    qpos_wrist_left = np.zeros((n, 7))
    qpos_finger_left = np.zeros((n, 5, 7))
    qpos_obj_left = np.zeros((n, 7))

    for i in range(n):
        qpos_wrist_right[i, :3] = r_global.apply(wrist_world[i])
        if full_skeleton_available:
            r = r_global * R.from_rotvec(wrist_rot_axisangle[i]) * r_right_wrist_offset
        else:
            # no orientation in fallback; use identity
            r = r_global * r_right_wrist_offset
        xyzw = r.as_quat()
        qpos_wrist_right[i, 3:] = np.concatenate([xyzw[3:], xyzw[:3]])

    for i in range(n):
        for j in range(5):
            qpos_finger_right[i, j, :3] = r_global.apply(tips_world[i, j])
            qpos_finger_right[i, j, 3:] = unit_quat

    for i in range(n):
        T_obj = obj_traj[i]
        qpos_obj_right[i, :3] = r_global.apply(T_obj[:3, 3])
        r = r_global * R.from_matrix(T_obj[:3, :3])
        xyzw = r.as_quat()
        qpos_obj_right[i, 3:] = np.concatenate([xyzw[3:], xyzw[:3]])

    qpos_wrist_left[:, 3:] = unit_quat
    qpos_finger_left[:, :, 3:] = unit_quat
    qpos_obj_left[:, 3:] = unit_quat

    # Compute the object's lowest world-frame z at frame 0 (where to place
    # the support plate, if any) AND the lowest z over the whole trajectory
    # across the object mesh + hand keypoints (where to place the world floor
    # so nothing dips below it). We DO NOT shift the trajectory — preserve raw
    # mocap coordinates.
    visual_mesh = pymeshlab.MeshSet()
    visual_mesh.load_new_mesh(f"{mesh_dir}/visual.obj")
    obj_verts_local = np.asarray(visual_mesh.current_mesh().vertex_matrix())
    R_obj0 = R.from_matrix(obj_traj[0][:3, :3])
    R_world = r_global * R_obj0
    obj_verts_world_z_frame0 = (
        R_world.apply(obj_verts_local) + qpos_obj_right[0, :3]
    ).min(axis=0)[2]
    # Sample object's lowest world z at every output frame and take the global min.
    obj_min_z_per_frame = []
    for k in range(n):
        Rk = r_global * R.from_matrix(obj_traj[k][:3, :3])
        zk = (Rk.apply(obj_verts_local) + qpos_obj_right[k, :3]).min(axis=0)[2]
        obj_min_z_per_frame.append(zk)
    obj_min_z_traj = float(np.min(obj_min_z_per_frame))
    scene_min_z = float(min(
        obj_min_z_traj,
        qpos_wrist_right[:, 2].min(),
        qpos_finger_right[:, :, 2].min(),
    ))
    # If the trajectory takes the object significantly below its frame-0
    # height (e.g. the cap of uncap_alcohol_burner moving down during the
    # uncap motion), a fixed support plate at frame-0 level would block
    # physics and clip through the floor. In that case, skip the plate and
    # rely on the world floor alone for support.
    object_descends = obj_min_z_traj < obj_verts_world_z_frame0 - 0.02
    task_info["obj_first_frame_lowest_world_z"] = float(obj_verts_world_z_frame0)
    task_info["scene_lowest_world_z"] = scene_min_z
    task_info["object_descends_from_frame0"] = bool(object_descends)
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(
        f"Object frame-0 lowest world z = {obj_verts_world_z_frame0:+.3f}m; "
        f"scene lowest world z = {scene_min_z:+.3f}m"
        + (" (object descends -- skip support plate)" if object_descends else "")
    )

    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")

    # 13. viser visualization
    if not show_viewer:
        return

    # Build mujoco scene with object mocap body so the existing viser viewer
    # can render the mesh; we drive the hand directly via the viser scene API.
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")
    object_right_handle = mj_spec.worldbody.add_body(name="right_object", mocap=True)
    object_right_handle.add_site(
        name="right_object", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.001, 0.001, 0.001], rgba=[1, 0, 0, 0], group=3,
    )
    mj_spec.add_mesh(
        name="right_object",
        file=f"{dataset_dir}/{task_info['right_object_mesh_dir']}/visual.obj",
    )
    object_right_handle.add_geom(
        name="right_object", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="right_object", pos=[0, 0, 0], quat=[1, 0, 0, 0],
        group=0, condim=1,
    )
    object_left_handle = mj_spec.worldbody.add_body(name="left_object", mocap=True)
    object_left_handle.add_site(
        name="left_object", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.001, 0.001, 0.001], rgba=[0, 1, 0, 0], group=3,
    )
    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)

    qpos_list = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )

    from spider.viewers.viser_viewer import (
        _STATE,
        build_and_log_scene_from_spec,
        init_viser,
        log_frame,
    )

    init_viser(app_name="spider-oakinkv2")
    body_entity_and_ids = build_and_log_scene_from_spec(
        mj_spec, mj_model, build_ref=False
    )

    # Build a separate hand-skeleton overlay (rotated into mujoco z-up).
    server = _STATE.server
    if full_skeleton_available:
        joints_mj = np.stack(
            [r_global.apply(joints_world_full[t]) for t in range(n)], axis=0
        )  # (n, 21, 3)
        # Add 21 spheres + 20 line bones, animate per frame via on_update slider.
        joint_handles = [
            server.scene.add_icosphere(
                f"hand/joint_{i}",
                radius=0.006,
                color=(220, 80, 80) if i in FINGERTIP_INDICES else (80, 130, 220),
                position=tuple(joints_mj[0, i]),
            )
            for i in range(21)
        ]
        bone_handles = []
        for parent, child in SKELETON_BONES:
            h = server.scene.add_spline_catmull_rom(
                f"hand/bone_{parent}_{child}",
                positions=np.stack([joints_mj[0, parent], joints_mj[0, child]]),
                color=(255, 200, 50),
                line_width=2.5,
            )
            bone_handles.append((h, parent, child))
    else:
        joints_mj = None
        n_fp = 5
        joint_handles = [
            server.scene.add_icosphere(
                f"hand/wrist",
                radius=0.012, color=(80, 130, 220),
                position=tuple(qpos_wrist_right[0, :3]),
            )
        ] + [
            server.scene.add_icosphere(
                f"hand/tip_{j}", radius=0.008, color=(220, 80, 80),
                position=tuple(qpos_finger_right[0, j, :3]),
            )
            for j in range(n_fp)
        ]
        bone_handles = []

    # Push frames via log_frame (records mujoco bodies including object mesh).
    loguru.logger.info(
        f"Viser scene built; pushing {n} frames. Open the printed URL "
        "in your browser."
    )
    for t in range(n):
        mj_data.mocap_pos[:] = qpos_list[t, :, :3]
        mj_data.mocap_quat[:] = qpos_list[t, :, 3:]
        mujoco.mj_kinematics(mj_model, mj_data)
        log_frame(
            mj_data, sim_time=t / target_fps,
            viewer_body_entity_and_ids=body_entity_and_ids,
            show_ui=True, playback_fps=target_fps,
        )

    # Wire the timeline slider to update hand keypoints/skeleton too.
    def _update_hand(frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, n - 1))
        if full_skeleton_available:
            for i, h in enumerate(joint_handles):
                h.position = tuple(joints_mj[frame_idx, i])
            for h, parent, child in bone_handles:
                h.positions = np.stack(
                    [joints_mj[frame_idx, parent], joints_mj[frame_idx, child]]
                )
        else:
            joint_handles[0].position = tuple(qpos_wrist_right[frame_idx, :3])
            for j in range(5):
                joint_handles[1 + j].position = tuple(qpos_finger_right[frame_idx, j, :3])

    if _STATE.playback_slider is not None:
        @_STATE.playback_slider.on_update
        def _(_):
            _update_hand(int(_STATE.playback_slider.value))

    loguru.logger.info(
        "All frames logged. Use the Timeline slider in the viser browser to inspect."
    )
    loguru.logger.info("Press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
