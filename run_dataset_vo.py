"""Run the released DPVO demo over every scene in a dataset."""

import argparse
import json
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from tqdm import tqdm

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.utils import Timer
from eval.trajectory_metrics import (
    evaluate_trajectory_files,
    format_trajectory_metrics,
    save_kitti_trajectory,
    save_trajectory_route_plot,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_NETWORK = PROJECT_ROOT / "models" / "dpvo.pth"
IMAGE_SUFFIXES = {".png", ".jpeg", ".jpg"}


# -----------------------------------------------------------------------------
# DPVO inference: kept from demo.py.
# Only two changes are made:
#   1. calib is the [fx, fy, cx, cy] vector read from JSON.
#   2. run() displays per-frame progress.
# -----------------------------------------------------------------------------

def image_stream(queue, imagedir, calib, stride, skip=0):
    """Image reader adapted from dpvo/stream.py for in-memory intrinsics."""
    fx, fy, cx, cy = calib
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    image_list = sorted(
        path
        for path in Path(imagedir).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if image is None:
            raise ValueError(f"OpenCV could not read {imfile}")

        height, width = image.shape[:2]
        image = image[: height - height % 16, : width - width % 16]
        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):
    slam = None
    queue = Queue(maxsize=8)

    reader = Process(
        target=image_stream,
        args=(queue, imagedir, calib, stride, skip),
    )
    reader.start()

    image_count = len([
        path
        for path in Path(imagedir).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    ][skip::stride])
    scene = Path(imagedir).parent.name
    progress = tqdm(total=image_count, desc=f"KITTI {scene}", unit="frame", dynamic_ncols=True)

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0:
            break

        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)
        progress.update()

    progress.close()
    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))


# -----------------------------------------------------------------------------
# New dataset/evaluation code.
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run DPVO on all scenes and save raw and Sim(3)-aligned results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--intrinsics-json", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="kitti")
    parser.add_argument("--image-subdir", default="image_2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--network", default=str(DEFAULT_NETWORK))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    gt_dir = args.gt_dir.resolve()
    output_dir = (args.output_root / args.dataset_name).resolve()

    with args.intrinsics_json.open("r", encoding="utf-8") as handle:
        intrinsics = json.load(handle)

    scenes = sorted(
        scene
        for scene in intrinsics
        if (dataset_root / scene).is_dir()
        and (gt_dir / f"{scene}.txt").is_file()
    )
    if not scenes:
        raise ValueError("No scenes have images, intrinsics, and ground truth")

    image_counts = {}
    for scene in scenes:
        image_dir = dataset_root / scene / args.image_subdir
        image_counts[scene] = sum(
            path.suffix.lower() in IMAGE_SUFFIXES
            for path in image_dir.iterdir()
        )
        if image_counts[scene] == 0:
            raise ValueError(f"No PNG/JPEG/JPG images found in {image_dir}")

        calib = np.asarray(intrinsics[scene], dtype=np.float32)
        if calib.shape != (4,) or not np.all(np.isfinite(calib)):
            raise ValueError(f"Invalid intrinsics for scene {scene}: {intrinsics[scene]}")

    before_dir = output_dir / "before_alignment"
    after_dir = output_dir / "after_alignment"
    before_plot_dir = output_dir / "visualizations" / "before_alignment"
    after_plot_dir = output_dir / "visualizations" / "after_alignment"
    for directory in (before_dir, after_dir, before_plot_dir, after_plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.LOOP_CLOSURE = False
    cfg.CLASSIC_LOOP_CLOSURE = False
    cfg.BUFFER_SIZE = max(cfg.BUFFER_SIZE, max(image_counts.values()) + 1)

    print(f"Dataset: {args.dataset_name}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Output: {output_dir}")
    print("Stride: 1")
    print("Loop closure: disabled")

    scene_metrics = {}
    for scene in scenes:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        image_dir = dataset_root / scene / args.image_subdir
        gt_path = gt_dir / f"{scene}.txt"
        raw_path = before_dir / f"{scene}.txt"
        aligned_path = after_dir / f"{scene}.txt"

        print(f"\n[{scene}] {image_counts[scene]} images")
        if raw_path.exists() and not args.overwrite:
            print(f"[{scene}] Reusing {raw_path}")
        else:
            (poses, timestamps), _ = run(
                cfg,
                args.network,
                image_dir,
                np.asarray(intrinsics[scene], dtype=np.float32),
                stride=1,
            )
            trajectory = PoseTrajectory3D(
                positions_xyz=poses[:, :3],
                orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
                timestamps=timestamps,
            )
            file_interface.write_kitti_poses_file(raw_path, trajectory)
            print(f"[{scene}] Saved {raw_path}")

        before_metrics, _, _ = evaluate_trajectory_files(
            gt_path,
            raw_path,
            alignment=None,
        )
        after_metrics, aligned_poses, sim3 = evaluate_trajectory_files(
            gt_path,
            raw_path,
            alignment="7dof",
        )

        save_trajectory_route_plot(
            before_dir,
            gt_dir,
            sequence=scene,
            method_name="DPVO (before alignment)",
            color="tab:orange",
            save_path=before_plot_dir / f"{scene}.png",
            dpi=args.plot_dpi,
        )

        if aligned_poses is None:
            sim3_result = None
            print(f"[{scene}] Sim(3) alignment unavailable: trajectory has no positional spread")
        else:
            assert aligned_poses is not None and sim3 is not None
            save_kitti_trajectory(aligned_poses, str(aligned_path))
            save_trajectory_route_plot(
                after_dir,
                gt_dir,
                sequence=scene,
                method_name="DPVO (Sim(3) aligned)",
                color="tab:blue",
                save_path=after_plot_dir / f"{scene}.png",
                dpi=args.plot_dpi,
            )
            sim3_result = {
                "scale": sim3["scale"],
                "rotation": sim3["rotation"].tolist(),
                "translation": sim3["translation"].tolist(),
            }

        scene_metrics[scene] = {
            "num_poses": image_counts[scene],
            "before_alignment": before_metrics,
            "after_alignment": after_metrics,
            "sim3": sim3_result,
        }
        print(format_trajectory_metrics(f"{scene} before alignment", before_metrics))
        print(format_trajectory_metrics(f"{scene} after Sim(3)", after_metrics))

        torch.cuda.empty_cache()

    metric_names = ("ave_translation", "ave_rotation", "ate", "ave_scale_error")
    averages = {}
    for alignment in ("before_alignment", "after_alignment"):
        averages[alignment] = {}
        for metric in metric_names:
            available_values = [
                scene_metrics[scene][alignment][metric]
                for scene in scenes
                if scene_metrics[scene][alignment][metric] is not None
            ]
            averages[alignment][metric] = (
                float(np.mean(available_values))
                if available_values
                else None
            )

    print("\nAggregate metrics")
    print(format_trajectory_metrics("Average before alignment", averages["before_alignment"]))
    print(format_trajectory_metrics("Average after Sim(3)", averages["after_alignment"]))

    report = {
        "dataset": args.dataset_name,
        "alignment": "Sim(3) / 7DoF (DPVO paper protocol)",
        "scenes": scene_metrics,
        "average": averages,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
