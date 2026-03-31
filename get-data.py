import argparse
import json
import os
from typing import Optional, Sequence

from crowtools.datasets import (
    ensure_dataset,
    ensure_default_datasets,
    get_dataset_artifact_path,
    get_dataset_libraries,
    get_selected_files,
    materialize_dataset_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare crow-tools data in a dataset-aware way."
    )
    parser.add_argument(
        "--dataset",
        default="starter",
        help="Dataset to build. Defaults to the small deterministic starter dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the cache directory. Defaults to ./.cache or $CROW_TOOLS_CACHE_DIR.",
    )
    parser.add_argument(
        "--include-backgrounds",
        action="store_true",
        help="Also download background audio used by separator training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without running any pipeline steps.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading audio.")
    parser.add_argument("--skip-denoise", action="store_true", help="Skip denoising.")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding.")
    parser.add_argument("--skip-detect", action="store_true", help="Skip detection.")
    return parser


def describe_plan(dataset_name: str, cache_dir: Optional[str], include_backgrounds: bool) -> str:
    config = ensure_dataset(dataset_name, cache_dir)
    libraries = get_dataset_libraries(dataset_name, cache_dir)
    selected_files = get_selected_files(dataset_name, cache_dir)

    lines = [
        f"Dataset: {dataset_name}",
        f"Cache: {os.path.abspath(cache_dir) if cache_dir else 'default'}",
        f"Libraries: {', '.join(libraries) if libraries else '(none)'}",
        f"Background downloads: {'enabled' if include_backgrounds else 'disabled'}",
    ]

    if selected_files:
        for library_name in libraries:
            file_ids = selected_files.get(library_name)
            count = len(file_ids) if file_ids is not None else "all"
            lines.append(f"Selected files for {library_name}: {count}")
    else:
        lines.append("Selected files: all files in included libraries")

    if config.get("description"):
        lines.append(f"Description: {config['description']}")
    return "\n".join(lines)


def resolve_pipeline_functions():
    from downloader.download_backgrounds import start_downloads as download_backgrounds
    from downloader.download_crows_ebird import start_downloads as download_macaulay
    from downloader.download_crows_xeno import start_downloads as download_xeno
    from denoiser.denoise_all import start_denoising
    from detector.detect_all import start_detections
    from embedder.embed_all import start_embeddings

    return {
        "download_backgrounds": download_backgrounds,
        "download_macaulay": download_macaulay,
        "download_xeno": download_xeno,
        "start_denoising": start_denoising,
        "start_detections": start_detections,
        "start_embeddings": start_embeddings,
    }


def has_curated_dataset_artifacts(dataset_name: str, cache_dir: Optional[str]) -> bool:
    if dataset_name != "starter":
        return False

    labels_path = get_dataset_artifact_path(dataset_name, "labels.json", cache_base=cache_dir)
    segments_path = get_dataset_artifact_path(dataset_name, "segments.json", cache_base=cache_dir)

    try:
        with open(labels_path, "r", encoding="utf-8") as handle:
            labels = json.load(handle)
        with open(segments_path, "r", encoding="utf-8") as handle:
            segments = json.load(handle)
    except FileNotFoundError:
        return False

    return bool(labels) and bool(segments)


def run_pipeline(dataset_name: str, cache_dir: Optional[str], include_backgrounds: bool, args) -> None:
    selected_files = get_selected_files(dataset_name, cache_dir)
    libraries = get_dataset_libraries(dataset_name, cache_dir)
    pipeline = resolve_pipeline_functions()
    curated_artifacts = has_curated_dataset_artifacts(dataset_name, cache_dir)

    if not args.skip_download:
        if "macaulay" in libraries:
            pipeline["download_macaulay"](
                selected_ids=selected_files.get("macaulay"),
                cache_base=cache_dir,
            )
        if "xeno-canto" in libraries:
            pipeline["download_xeno"](
                selected_ids=selected_files.get("xeno-canto"),
                cache_base=cache_dir,
            )
        if include_backgrounds:
            pipeline["download_backgrounds"](cache_base=cache_dir)

    if not args.skip_denoise:
        pipeline["start_denoising"](
            libraries=libraries,
            selected_ids_by_library=selected_files,
            cache_base=cache_dir,
        )

    if not args.skip_embed:
        pipeline["start_embeddings"](
            denoised=False,
            libraries=libraries,
            selected_ids_by_library=selected_files,
            cache_base=cache_dir,
        )
        pipeline["start_embeddings"](
            denoised=True,
            libraries=libraries,
            selected_ids_by_library=selected_files,
            cache_base=cache_dir,
        )

    if not args.skip_detect and not curated_artifacts:
        pipeline["start_detections"](
            libraries=libraries,
            selected_ids_by_library=selected_files,
            cache_base=cache_dir,
        )

    if curated_artifacts:
        print(f"Using curated dataset labels and segments for {dataset_name}; skipping auto-label materialization.")
    else:
        materialize_dataset_artifacts(dataset_name, cache_base=cache_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_default_datasets(args.cache_dir)
    ensure_dataset(args.dataset, args.cache_dir)

    print(describe_plan(args.dataset, args.cache_dir, args.include_backgrounds))
    if args.dry_run:
        return 0

    run_pipeline(args.dataset, args.cache_dir, args.include_backgrounds, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
