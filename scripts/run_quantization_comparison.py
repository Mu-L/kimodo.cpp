#!/usr/bin/env python3
"""Create a reproducible, viewer-ready Kimodo text-quantization comparison."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import quantization_metrics


LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COLORS = ("#65eee1", "#ffbd4a", "#d783ff", "#ff657a", "#8da2ff", "#7ce38b")


def variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be LABEL=TEXT_BUNDLE")
    label, path = value.split("=", 1)
    if not LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError("variant label contains unsafe characters")
    bundle = Path(path).resolve()
    if not bundle.is_dir():
        raise argparse.ArgumentTypeError(f"text bundle is not a directory: {bundle}")
    return label, bundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], environment: dict[str, str]) -> float:
    started = time.monotonic()
    subprocess.run(command, check=True, env=environment)
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--reference", type=variant, required=True, metavar="LABEL=BUNDLE")
    parser.add_argument("--variant", type=variant, action="append", default=[], metavar="LABEL=BUNDLE")
    parser.add_argument("--skeleton", choices=("smplx22", "soma30", "g1skel34"), required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=Path, help="captured F32 noise; generated with NumPy PCG64 when omitted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=Path("build/debug/kmd-encode"))
    parser.add_argument("--sampler", type=Path, default=Path("build/debug/kmd-sample-embedding"))
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--jobs", type=int, default=1, help="independent variants to run concurrently")
    parser.add_argument("--threads", type=int, help="KIMODO_THREADS per variant")
    args = parser.parse_args()
    if args.frames < 2 or args.steps < 1:
        parser.error("frames must be at least 2 and steps must be positive")
    if args.jobs < 1 or (args.threads is not None and args.threads < 1):
        parser.error("jobs and threads must be positive")
    all_variants = [args.reference, *args.variant]
    if len({label for label, _ in all_variants}) != len(all_variants):
        parser.error("variant labels must be unique")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text().strip()
    if not prompt:
        parser.error("prompt is empty")
    shutil.copyfile(args.prompt, output / "prompt.txt")

    dimensions = {"smplx22": 273, "soma30": 369, "g1skel34": 417}
    noise_path = output / "initial_noise.f32"
    if args.noise:
        noise = np.fromfile(args.noise, dtype="<f4")
        if noise.size != args.frames * dimensions[args.skeleton] or not np.isfinite(noise).all():
            parser.error("captured noise dimensions or values are invalid")
        shutil.copyfile(args.noise, noise_path)
    else:
        noise = np.random.Generator(np.random.PCG64(args.seed)).standard_normal(
            args.frames * dimensions[args.skeleton], dtype=np.float32)
        noise.astype("<f4", copy=False).tofile(noise_path)

    environment = os.environ.copy()
    environment["KIMODO_BACKEND"] = args.backend
    if args.threads is not None:
        environment["KIMODO_THREADS"] = str(args.threads)

    def process(item: tuple[int, tuple[str, Path]]) -> dict[str, object]:
        index, (label, bundle) = item
        directory = output / label
        directory.mkdir()
        embedding = directory / "embedding.f32"
        encode_seconds = run([str(args.encoder), str(bundle), str(args.prompt), str(embedding)], environment)
        sample_seconds = run([
            str(args.sampler), str(args.motion_model), str(embedding), str(noise_path),
            str(args.frames), str(args.steps), str(directory),
        ], environment)
        return {
            "label": label,
            "color": COLORS[index % len(COLORS)],
            "bundle": str(bundle),
            "bundle_bytes": sum(path.stat().st_size for path in bundle.glob("*.gguf")),
            "embedding_sha256": sha256(embedding),
            "encode_seconds": encode_seconds,
            "sample_seconds": sample_seconds,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        records = list(executor.map(process, enumerate(all_variants)))

    reference_label = args.reference[0]
    reference_dir = output / reference_label
    metrics = {
        label: quantization_metrics.compare(reference_dir, output / label)
        for label, _ in args.variant
    }
    skeleton = json.loads((reference_dir / "skeleton.json").read_text())
    comparison = {
        "schema_version": 1,
        "id": output.name,
        "prompt": prompt,
        "prompt_sha256": sha256(args.prompt),
        "motion_model": str(args.motion_model.resolve()),
        "motion_model_sha256": sha256(args.motion_model),
        "skeleton": skeleton,
        "frames": args.frames,
        "steps": args.steps,
        "seed": args.seed,
        "backend": args.backend,
        "jobs": args.jobs,
        "threads_per_variant": args.threads,
        "noise_sha256": sha256(noise_path),
        "reference": reference_label,
        "variants": records,
        "metrics": metrics,
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"wrote {output / 'comparison.json'} with {len(records)} synchronized variants")


if __name__ == "__main__":
    main()
