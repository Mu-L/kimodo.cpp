#!/usr/bin/env python3
"""Measure paired embedding and motion divergence against a reference run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_f32(path: Path) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"invalid or non-finite F32 data: {path}")
    return value.astype(np.float64)


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    delta = candidate - reference
    denom = float(np.linalg.norm(reference))
    cosine = float(np.dot(reference, candidate) / (denom * np.linalg.norm(candidate)))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return {
        "cosine_similarity": cosine,
        "cosine_distance": 1.0 - cosine,
        "angular_error_degrees": math.degrees(math.acos(cosine)),
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "norm_ratio": float(np.linalg.norm(candidate) / denom),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_absolute": float(np.mean(np.abs(delta))),
        "max_absolute": float(np.max(np.abs(delta))),
    }


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), axis=-1)


def rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., :3]
    t = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[..., 3:4] * t + np.cross(xyz, t)


def forward_kinematics(roots: np.ndarray, local: np.ndarray, parents: list[int], offsets: np.ndarray) -> np.ndarray:
    frames, joints, _ = local.shape
    norm = np.linalg.norm(local, axis=-1, keepdims=True)
    local = local / np.maximum(norm, 1e-12)
    world_rotation = np.empty_like(local)
    positions = np.empty((frames, joints, 3), dtype=np.float64)
    for joint, parent in enumerate(parents):
        if parent < 0:
            world_rotation[:, joint] = local[:, joint]
            positions[:, joint] = roots
        else:
            world_rotation[:, joint] = quaternion_multiply(world_rotation[:, parent], local[:, joint])
            positions[:, joint] = positions[:, parent] + rotate(
                world_rotation[:, parent], np.broadcast_to(offsets[joint], (frames, 3)))
    return positions


def motion_metrics(reference_dir: Path, candidate_dir: Path) -> dict[str, object]:
    skeleton = json.loads((reference_dir / "skeleton.json").read_text())
    candidate_skeleton = json.loads((candidate_dir / "skeleton.json").read_text())
    for field in ("key", "frames", "fps", "parents", "offsets"):
        if skeleton.get(field) != candidate_skeleton.get(field):
            raise ValueError(f"skeleton mismatch in {field}")
    frames, parents = int(skeleton["frames"]), skeleton["parents"]
    joints = len(parents)
    offsets = np.asarray(skeleton["offsets"], dtype=np.float64)
    ref_root = read_f32(reference_dir / "root_positions.f32").reshape(frames, 3)
    got_root = read_f32(candidate_dir / "root_positions.f32").reshape(frames, 3)
    ref_rot = read_f32(reference_dir / "local_rotations_xyzw.f32").reshape(frames, joints, 4)
    got_rot = read_f32(candidate_dir / "local_rotations_xyzw.f32").reshape(frames, joints, 4)
    ref_rot /= np.maximum(np.linalg.norm(ref_rot, axis=-1, keepdims=True), 1e-12)
    got_rot /= np.maximum(np.linalg.norm(got_rot, axis=-1, keepdims=True), 1e-12)
    root_error = np.linalg.norm(got_root - ref_root, axis=-1)
    horizontal_error = np.linalg.norm((got_root - ref_root)[:, (0, 2)], axis=-1)
    dots = np.clip(np.abs(np.sum(ref_rot * got_rot, axis=-1)), 0.0, 1.0)
    rotation_degrees = np.degrees(2.0 * np.arccos(dots))
    ref_world = forward_kinematics(ref_root, ref_rot, parents, offsets)
    got_world = forward_kinematics(got_root, got_rot, parents, offsets)
    world_error = np.linalg.norm(got_world - ref_world, axis=-1)
    ref_relative = ref_world - ref_root[:, None, :]
    got_relative = got_world - got_root[:, None, :]
    relative_error = np.linalg.norm(got_relative - ref_relative, axis=-1)
    ref_velocity, got_velocity = np.diff(ref_world, axis=0), np.diff(got_world, axis=0)
    velocity_error = np.linalg.norm(got_velocity - ref_velocity, axis=-1) * float(skeleton["fps"])
    raw_ref = read_f32(reference_dir / "sampling_final_state.f32")
    raw_got = read_f32(candidate_dir / "sampling_final_state.f32")
    return {
        "raw_final_state": vector_metrics(raw_ref, raw_got),
        "root_position_cm": distribution(root_error * 100.0),
        "root_horizontal_cm": distribution(horizontal_error * 100.0),
        "root_endpoint_cm": float(root_error[-1] * 100.0),
        "joint_rotation_degrees": distribution(rotation_degrees),
        "world_mpjpe_cm": distribution(world_error * 100.0),
        "root_relative_mpjpe_cm": distribution(relative_error * 100.0),
        "joint_velocity_error_cm_per_second": distribution(velocity_error * 100.0),
        "per_frame_world_mpjpe_cm": (np.mean(world_error, axis=1) * 100.0).tolist(),
        "per_frame_root_relative_mpjpe_cm": (np.mean(relative_error, axis=1) * 100.0).tolist(),
    }


def compare(reference_dir: Path, candidate_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "embedding": vector_metrics(
            read_f32(reference_dir / "embedding.f32"),
            read_f32(candidate_dir / "embedding.f32"),
        )
    }
    if (reference_dir / "root_positions.f32").exists() and (candidate_dir / "root_positions.f32").exists():
        result["motion"] = motion_metrics(reference_dir, candidate_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("LABEL", "DIRECTORY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "reference": str(args.reference),
        "candidates": {label: compare(args.reference, Path(directory)) for label, directory in args.candidate},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output} with {len(report['candidates'])} candidate comparison(s)")


if __name__ == "__main__":
    main()
