from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from framework.eval.vo_eval_utils import KittiEvalOdom
from framework.utils.plotting import get_pyplot


CameraMotion = Tuple[np.ndarray, np.ndarray]
CameraMotionPredictions = Dict[str, Sequence[CameraMotion]]
NedMotionPredictions = Dict[str, Sequence[np.ndarray]]
PathLike = Union[str, Path]
NED_TO_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def rel_pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.astype(np.float64)
    transform[:3, 3] = translation.astype(np.float64)
    return transform


def pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose vector shape (6,), got {pose.shape}")
    return rel_pose_matrix(R.from_rotvec(pose[3:6]).as_matrix(), pose[:3])


def convert_ned_pose_to_camera(pose_ned: np.ndarray) -> np.ndarray:
    pose_cam = np.eye(4, dtype=np.float64)
    pose_cam[:3, :3] = NED_TO_CAMERA @ pose_ned[:3, :3] @ NED_TO_CAMERA.T
    pose_cam[:3, 3] = NED_TO_CAMERA @ pose_ned[:3, 3]
    return pose_cam


def accumulate_relative_motions(motions: Sequence[np.ndarray]) -> List[np.ndarray]:
    current_pose = np.eye(4, dtype=np.float64)
    poses = [current_pose.copy()]
    for motion in motions:
        current_pose = current_pose @ motion
        poses.append(current_pose.copy())
    return poses


def save_kitti_trajectory(poses: Sequence[np.ndarray], path: str) -> None:
    arr = np.stack([pose[:3, :].reshape(12) for pose in poses], axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, arr, fmt="%.9f")


def save_gt_trajectory_subset(pose_path: PathLike, num_frames: int, save_path: PathLike) -> None:
    poses = load_kitti_trajectory(pose_path)
    if num_frames > len(poses):
        raise ValueError(f"Requested {num_frames} GT frames from {pose_path}, but only {len(poses)} are available.")
    save_kitti_trajectory([_pose_row_to_matrix(row) for row in poses[:num_frames]], str(save_path))


def _pose_row_to_matrix(row: np.ndarray) -> np.ndarray:
    pose = np.asarray(row, dtype=np.float64).reshape(-1)
    if pose.size < 12:
        raise ValueError(f"Expected at least 12 pose values, got {pose.size}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :] = pose[:12].reshape(3, 4)
    return transform


def load_kitti_trajectory(path: PathLike) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pose file not found: {path}")

    poses = np.loadtxt(path, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] < 12:
        raise ValueError(f"Unexpected pose format in {path}. Expected Nx12+, got {poses.shape}.")
    return poses


def compute_average_trajectory_metrics(
    gt_dir: PathLike,
    pred_dir: PathLike,
    *,
    alignment: Optional[str] = None,
) -> Dict[str, float]:
    evaluator = KittiEvalOdom()
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    metric_sums = {
        "ave_translation": 0.0,
        "ave_rotation": 0.0,
        "ate": 0.0,
        "ave_scale_error": 0.0,
    }
    num_sequences = 0

    for pred_path in sorted(pred_dir.glob("*.txt")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT trajectory for prediction {pred_path.name}: {gt_path}")

        poses_gt = evaluator.load_poses_from_txt(str(gt_path))
        poses_pred = evaluator.load_poses_from_txt(str(pred_path))
        seq_err = evaluator.calc_sequence_errors(poses_gt, poses_pred)
        ave_t_err, ave_r_err = evaluator.compute_overall_err(seq_err)

        metric_sums["ave_translation"] += float(ave_t_err * 100.0)
        metric_sums["ave_rotation"] += float(ave_r_err / np.pi * 180.0 * 100.0)
        metric_sums["ate"] += float(evaluator.compute_ATE(poses_gt, poses_pred))
        metric_sums["ave_scale_error"] += float(evaluator.compute_ASE(poses_gt, poses_pred))
        num_sequences += 1

    if num_sequences == 0:
        return {}

    return {key: value / num_sequences for key, value in metric_sums.items()}


def extract_xz_coordinates(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return poses[:, 3], poses[:, 11]


def clear_prediction_dir(save_dir: str) -> None:
    """Remove previously written trajectory `.txt` predictions from a directory."""
    os.makedirs(save_dir, exist_ok=True)
    for name in os.listdir(save_dir):
        if name.endswith(".txt"):
            os.remove(os.path.join(save_dir, name))


def save_camera_motion_predictions(
    predictions: CameraMotionPredictions,
    save_dir: str,
) -> None:
    """Save camera-frame relative motions as trajectory files in KITTI pose format."""
    clear_prediction_dir(save_dir)

    for scene_id in sorted(predictions.keys()):
        camera_motions = predictions[scene_id]
        abs_poses = accumulate_relative_motions(
            [rel_pose_matrix(rotation, translation) for rotation, translation in camera_motions]
        )
        save_kitti_trajectory(abs_poses, os.path.join(save_dir, f"{scene_id}.txt"))


def save_ned_motion_predictions(
    predictions: NedMotionPredictions,
    save_dir: str,
) -> None:
    """Save NED-frame 6D motion vectors as trajectory files in KITTI pose format."""
    clear_prediction_dir(save_dir)

    for scene_id in sorted(predictions.keys()):
        motion_mats = [pose_vector_to_matrix(pose) for pose in predictions[scene_id]]
        abs_poses_ned = accumulate_relative_motions(motion_mats)
        abs_poses_cam = [convert_ned_pose_to_camera(pose_ned) for pose_ned in abs_poses_ned]
        save_kitti_trajectory(abs_poses_cam, os.path.join(save_dir, f"{scene_id}.txt"))


def save_trajectory_route_plot(
    pred_dir: PathLike,
    gt_dir: PathLike,
    *,
    sequence: str,
    method_name: str,
    color: str = "b",
    linestyle: str = "--",
    save_path: Optional[PathLike] = None,
    dpi: int = 150,
) -> str:
    """Save an X-Z trajectory plot for one sequence."""
    plt = get_pyplot()

    pred_path = Path(pred_dir) / f"{sequence}.txt"
    gt_path = Path(gt_dir) / f"{sequence}.txt"

    gt_poses = load_kitti_trajectory(gt_path)
    pred_poses = load_kitti_trajectory(pred_path)
    x_gt, z_gt = extract_xz_coordinates(gt_poses)
    x_pred, z_pred = extract_xz_coordinates(pred_poses)

    fig, ax = plt.subplots(figsize=(10, 10))
    if len(x_gt) > 0:
        ax.scatter(x_gt[0], z_gt[0], label="Start", marker="s", color="k", zorder=3)
    ax.plot(x_gt, z_gt, color="k", label="Ground truth", linewidth=2.5)
    ax.plot(x_pred, z_pred, color=color, linestyle=linestyle, label=method_name, linewidth=3.0)
    ax.set_title(f"Visual Odometry - Sequence {sequence}", fontsize=18)
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Z [m]", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle="-", alpha=0.2)
    ax.legend(loc="upper left", fontsize="large")

    route_path = Path(save_path) if save_path is not None else (Path(pred_dir) / f"route_{sequence}.png")
    route_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(route_path, dpi=dpi)
    plt.close(fig)
    return str(route_path)


def save_trajectory_route_plots(
    pred_dir: PathLike,
    gt_dir: PathLike,
    *,
    method_name: str,
    color: str = "b",
    linestyle: str = "--",
    dpi: int = 150,
) -> List[str]:
    """Save route plots for every prediction file in a directory."""
    pred_dir = Path(pred_dir)
    saved_paths: List[str] = []
    for pred_path in sorted(pred_dir.glob("*.txt")):
        saved_paths.append(
            save_trajectory_route_plot(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                sequence=pred_path.stem,
                method_name=method_name,
                color=color,
                linestyle=linestyle,
                dpi=dpi,
            )
        )
    return saved_paths
