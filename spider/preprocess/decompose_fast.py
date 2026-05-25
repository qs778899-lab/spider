# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    raise SystemExit(1)


import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as _ScipyR

import spider


def _R_from_quat(q_xyzw: np.ndarray) -> np.ndarray:
    return _ScipyR.from_quat(q_xyzw).as_matrix()

MeshPart = tuple[np.ndarray, np.ndarray]


def fast_voxel_convex_decomp_from_pointcloud(
    points: np.ndarray, pitch: float = 0.1, min_points: int = 20
) -> list[MeshPart]:
    """Approximate convex decomposition via voxel clusters and convex hulls."""
    coords = np.floor(points / pitch).astype(int)
    unique_voxels, inverse = np.unique(coords, axis=0, return_inverse=True)

    hulls: list[MeshPart] = []
    for idx, _ in enumerate(unique_voxels):
        cluster_points = points[inverse == idx]
        if len(cluster_points) < min_points:
            continue

        cluster_mesh = trimesh.Trimesh(vertices=cluster_points, faces=[])
        hull = cluster_mesh.convex_hull
        vertices = np.asarray(hull.vertices)
        faces = np.asarray(hull.faces, dtype=int)
        hulls.append((vertices, faces))

    return hulls


def flatten_base(
    hulls: Iterable[MeshPart],
    thickness: float = 0.01,
    R_world_local: np.ndarray | None = None,
    obj_world_pos: np.ndarray | None = None,
    floor_z: float = 0.0,
    pad: float = 0.05,
    well_below_offset: float = 0.0,
) -> list[MeshPart]:
    """Append a thin plate that supports the object resting on a world-frame floor.

    If ``R_world_local`` and ``obj_world_pos`` are given, the plate is added in
    the object's *local* frame so that, when the free joint is set to
    (obj_world_pos, R_world_local), the plate sits *above* the world floor at
    ``floor_z`` (its bottom face just above the floor, top face supporting the
    object's lowest convex hulls). This guarantees the object does NOT drop in
    the first physics frame and avoids overlap between the plate and the world
    floor that would push the object up.

    Otherwise (the legacy behavior), the plate is placed at the convex hull's
    local-frame ``min_z``.

    The plate's XY extent is the object's world-frame bbox padded by ``pad``
    so it is large enough to support the object even if it tilts slightly.
    """
    hull_list = list(hulls)
    if not hull_list:
        return hull_list

    all_vertices = np.vstack([vertices for vertices, _ in hull_list])

    if R_world_local is not None and obj_world_pos is not None:
        # Compute world-frame XY footprint of the object.
        v_world = (R_world_local @ all_vertices.T).T + obj_world_pos
        wx_min, wx_max = v_world[:, 0].min() - pad, v_world[:, 0].max() + pad
        wy_min, wy_max = v_world[:, 1].min() - pad, v_world[:, 1].max() + pad
        # Place the plate so its TOP is at ``floor_z - well_below_offset``
        # (i.e. well below the object's lowest world-frame z) and the plate
        # extends ``thickness`` downward from there. The plate is body-fixed
        # to the object and follows it through the trajectory, but its low
        # offset means the plate sits well clear of the trajectory motion in
        # world frame and doesn't interfere with the object's natural path.
        z_top = floor_z - well_below_offset
        z_bot = z_top - thickness
        corners_world_top = np.array(
            [
                [wx_min, wy_min, z_top],
                [wx_max, wy_min, z_top],
                [wx_max, wy_max, z_top],
                [wx_min, wy_max, z_top],
            ]
        )
        corners_world_bot = corners_world_top.copy()
        corners_world_bot[:, 2] = z_bot
        corners_world = np.vstack([corners_world_bot, corners_world_top])
        # Transform world -> local: v_local = R^T (v_world - obj_world_pos)
        plate_vertices = (R_world_local.T @ (corners_world - obj_world_pos).T).T
    else:
        min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
        min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
        min_z = np.min(all_vertices[:, 2])
        z0 = min_z
        z1 = min_z + thickness
        plate_vertices = np.array(
            [
                [min_x, min_y, z0],
                [max_x, min_y, z0],
                [max_x, max_y, z0],
                [min_x, max_y, z0],
                [min_x, min_y, z1],
                [max_x, min_y, z1],
                [max_x, max_y, z1],
                [min_x, max_y, z1],
            ]
        )

    plate_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )

    hull_list.append((plate_vertices, plate_faces))
    return hull_list


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    add_floor: bool = False,
    floor_well_below_offset: float = 0.05,
    check_stability: bool = False,
    stability_max_drift: float = 0.005,
) -> None:
    dataset_path = Path(dataset_dir)

    if embodiment_type == "right":
        hands = ["right"]
    elif embodiment_type == "left":
        hands = ["left"]
    elif embodiment_type == "bimanual":
        hands = ["right", "left"]
    else:
        raise ValueError(f"Invalid hand type: {embodiment_type}")

    processed_dir = (
        dataset_path
        / "processed"
        / dataset_name
        / "mano"
        / embodiment_type
        / task
        / str(data_id)
    )
    task_info_path = processed_dir.parent / "task_info.json"

    if not task_info_path.exists():
        logger.error(
            "Missing task_info at {}. Run dataset preprocessing first.",
            task_info_path,
        )
        return

    with task_info_path.open("r", encoding="utf-8") as file:
        task_info = json.load(file)

    # Load first-frame object qpos from the trajectory_keypoints.npz so the
    # support plate can be aligned to the world-frame floor.
    keypoints_path = processed_dir / "trajectory_keypoints.npz"
    first_obj_pose: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "right": None,
        "left": None,
    }
    if keypoints_path.exists():
        kp = np.load(keypoints_path)
        for h in ("right", "left"):
            key = f"qpos_obj_{h}"
            if key in kp.files and len(kp[key]) > 0:
                pos = kp[key][0, :3]
                q_wxyz = kp[key][0, 3:]
                # only keep if the pose looks valid (not a zero placeholder)
                if not (np.allclose(pos, 0) and np.allclose(q_wxyz, [1, 0, 0, 0])):
                    q_xyzw = np.concatenate([q_wxyz[1:], q_wxyz[:1]])
                    R_obj = np.asarray(_R_from_quat(q_xyzw))
                    first_obj_pose[h] = (pos, R_obj)

    for hand in hands:
        mesh_dir_key = (
            "right_object_mesh_dir" if hand == "right" else "left_object_mesh_dir"
        )
        mesh_dir = task_info.get(mesh_dir_key)
        mesh_dir = f"{dataset_path}/{mesh_dir}"
        if not mesh_dir:
            logger.warning("No mesh_dir for {} hand; skipping.", hand)
            continue

        mesh_path = Path(mesh_dir)
        input_file = mesh_path / "visual.obj"
        output_dir = mesh_path / "convex"

        if not input_file.exists():
            logger.warning(
                "Input mesh {} does not exist. Skipping {} hand.", input_file, hand
            )
            continue

        mesh = trimesh.load(
            str(input_file), force="mesh", process=False, skip_materials=True
        )

        hulls = fast_voxel_convex_decomp_from_pointcloud(np.asarray(mesh.vertices))
        if not hulls:
            logger.warning("No convex parts generated for {}; skipping export.", hand)
            continue

        if add_floor:
            pose = first_obj_pose.get(hand)
            if pose is not None:
                obj_pos, R_obj = pose
                # Place the plate flush with the object's lowest world-frame
                # vertex at first-frame so the object rests stably without
                # dropping or hovering. The world floor sits 1mm below the
                # plate so they never overlap.
                hull_verts = np.vstack([v for v, _ in hulls])
                v_world = (R_obj @ hull_verts.T).T + obj_pos
                obj_min_world_z = float(v_world[:, 2].min())
                hulls = flatten_base(
                    hulls,
                    R_world_local=R_obj,
                    obj_world_pos=obj_pos,
                    floor_z=obj_min_world_z,
                    well_below_offset=floor_well_below_offset,
                )
                plate_top_z = obj_min_world_z - floor_well_below_offset
                task_info[f"{hand}_plate_top_world_z"] = float(plate_top_z)
                task_info["floor_well_below_offset"] = float(floor_well_below_offset)
                logger.info(
                    "Added support plate for {} hand: top at world z={:.4f} "
                    "({:.2f}m below object's frame-0 bottom).",
                    hand, plate_top_z, floor_well_below_offset,
                )
            else:
                hulls = flatten_base(hulls)
                logger.warning(
                    "No first-frame obj pose for {} hand; falling back to "
                    "local-frame plate (object may not rest stably on floor).",
                    hand,
                )
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (vertices, faces) in enumerate(hulls):
            mesh_part = trimesh.Trimesh(vertices, faces)
            part_path = output_dir / f"{idx}.obj"
            mesh_part.export(part_path)
            logger.info("Exported mesh part {} to {}", idx, part_path)

        convex_key = (
            "right_object_convex_dir" if hand == "right" else "left_object_convex_dir"
        )
        # get relative path to dataset_dir
        relative_path = os.path.relpath(output_dir, dataset_path)
        task_info[convex_key] = str(relative_path)

    with task_info_path.open("w", encoding="utf-8") as file:
        json.dump(task_info, file, indent=2)

    logger.info("Updated task_info with convex dirs at {}", task_info_path)

    if check_stability:
        for hand in hands:
            pose = first_obj_pose.get(hand)
            if pose is None:
                logger.warning(
                    "Skipping stability check for {} hand: no first-frame obj pose.",
                    hand,
                )
                continue
            convex_key = (
                "right_object_convex_dir"
                if hand == "right"
                else "left_object_convex_dir"
            )
            convex_dir = task_info.get(convex_key)
            if convex_dir is None:
                logger.warning(
                    "Skipping stability check for {} hand: no convex dir.", hand,
                )
                continue
            convex_path = dataset_path / convex_dir
            plate_top_z = task_info.get(f"{hand}_plate_top_world_z")
            if plate_top_z is not None:
                floor_z = float(plate_top_z) - 0.01 - 0.001
            else:
                obj_min_z = task_info.get("obj_first_frame_lowest_world_z")
                if obj_min_z is not None:
                    floor_z = float(obj_min_z) - 0.01 - 0.001
                else:
                    floor_z = 0.0
            ok, drift = _check_initial_stability(
                convex_path, pose[0], pose[1],
                floor_z=floor_z, max_drift=stability_max_drift,
            )
            if not ok:
                raise RuntimeError(
                    f"Stability check failed for {hand} object: it drifted "
                    f"{drift:.4f}m in 0.5s of physics (threshold "
                    f"{stability_max_drift}m). The support plate is likely "
                    "misaligned. Check obj_first_frame_pos in task_info.json."
                )
            logger.info(
                "Stability check passed for {} hand: drift={:.4f}m (limit {:.4f}m)",
                hand, drift, stability_max_drift,
            )


def _check_initial_stability(
    convex_dir: Path,
    obj_pos: np.ndarray,
    R_obj: np.ndarray,
    floor_z: float = 0.0,
    max_drift: float = 0.005,
    sim_dt: float = 0.005,
    sim_seconds: float = 0.5,
) -> tuple[bool, float]:
    """Drop-test the object at its first-frame pose on a floor at ``floor_z``.

    Builds a minimal mujoco scene with a floor + the object's convex hulls as
    a free body, sets the body to (obj_pos, R_obj), and steps physics. Returns
    (passed, max_drift_m).
    """
    import mujoco
    from scipy.spatial.transform import Rotation as R

    spec = mujoco.MjSpec()
    spec.option.timestep = sim_dt
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, floor_z],
    )
    body = spec.worldbody.add_body(name="obj", pos=[0, 0, 0])
    body.add_freejoint(name="obj_free")
    convex_files = sorted(convex_dir.glob("*.obj"))
    for i, fp in enumerate(convex_files):
        spec.add_mesh(name=f"convex_{i}", file=str(fp))
        body.add_geom(
            name=f"convex_{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"convex_{i}",
            condim=3,
            contype=1,
            conaffinity=1,
            density=400,
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    # Set free joint to first-frame world pose
    quat_xyzw = R.from_matrix(R_obj).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[3:], quat_xyzw[:3]])
    data.qpos[:3] = obj_pos
    data.qpos[3:7] = quat_wxyz
    mujoco.mj_forward(model, data)
    initial_pos = data.qpos[:3].copy()
    n_steps = int(sim_seconds / sim_dt)
    max_drift_seen = 0.0
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        drift = float(np.linalg.norm(data.qpos[:3] - initial_pos))
        if drift > max_drift_seen:
            max_drift_seen = drift
    return max_drift_seen <= max_drift, max_drift_seen


if __name__ == "__main__":
    tyro.cli(main)
