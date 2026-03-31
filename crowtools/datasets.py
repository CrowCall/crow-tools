import csv
import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set


PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
CACHE_ENV_VAR = "CROW_TOOLS_CACHE_DIR"

LOCAL_LIBRARY = "local"
BACKGROUND_LIBRARY = "backgrounds"
DEFAULT_PUBLIC_LIBRARIES = ["macaulay", "xeno-canto"]

# A deterministic, public-only starter subset intended for onboarding and smoke tests.
STARTER_SELECTED_FILES = {
    "macaulay": [
        "104365511",
        "229089",
        "56781051",
        "420980621",
        "304195801",
        "305346371",
        "391813591",
        "305970",
    ],
    "xeno-canto": [
        "543339",
        "543338",
        "543337",
        "543336",
        "543335",
        "524251",
        "524250",
        "524249",
    ],
}

DEFAULT_DATASET_CONFIGS = {
    "all-public": {
        "name": "all-public",
        "description": "All public crow audio libraries.",
        "included_libraries": DEFAULT_PUBLIC_LIBRARIES,
    },
    "starter": {
        "name": "starter",
        "description": "Small deterministic public subset for onboarding and smoke tests.",
        "included_libraries": DEFAULT_PUBLIC_LIBRARIES,
        "selected_files": STARTER_SELECTED_FILES,
    },
    "Local": {
        "name": "Local",
        "description": "Local-only dataset.",
        "included_libraries": [LOCAL_LIBRARY],
    },
}


def get_cache_base(cache_base: Optional[str] = None) -> str:
    return os.path.abspath(cache_base or os.environ.get(CACHE_ENV_VAR) or DEFAULT_CACHE_DIR)


def get_datasets_base(cache_base: Optional[str] = None) -> str:
    return os.path.join(get_cache_base(cache_base), "datasets")


def get_libraries_base(cache_base: Optional[str] = None) -> str:
    return os.path.join(get_cache_base(cache_base), "libraries")


def get_dataset_dir(dataset_name: str, cache_base: Optional[str] = None) -> str:
    return os.path.join(get_datasets_base(cache_base), dataset_name)


def get_dataset_artifact_path(dataset_name: str, *parts: str, cache_base: Optional[str] = None) -> str:
    return os.path.join(get_dataset_dir(dataset_name, cache_base), *parts)


def get_library_dir(library_name: str, cache_base: Optional[str] = None) -> str:
    return os.path.join(get_libraries_base(cache_base), library_name)


def read_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def merge_dataset_config(base: dict, override: Optional[dict]) -> dict:
    merged = dict(base or {})
    if override:
        for key, value in override.items():
            merged[key] = value

    included = merged.get("included_libraries", [])
    merged["included_libraries"] = list(dict.fromkeys(included))

    selected = merged.get("selected_files") or {}
    normalized_selected = {}
    for library_name, file_ids in selected.items():
        normalized_selected[library_name] = [str(file_id) for file_id in file_ids]
    if normalized_selected:
        merged["selected_files"] = normalized_selected
    return merged


def load_dataset_config(dataset_name: str, cache_base: Optional[str] = None) -> dict:
    config_path = os.path.join(get_dataset_dir(dataset_name, cache_base), "config.json")
    disk_config = read_json_file(config_path, None)
    default_config = DEFAULT_DATASET_CONFIGS.get(dataset_name, {"name": dataset_name})
    return merge_dataset_config(default_config, disk_config)


def ensure_dataset(dataset_name: str, cache_base: Optional[str] = None) -> dict:
    config = load_dataset_config(dataset_name, cache_base)
    dataset_dir = get_dataset_dir(dataset_name, cache_base)
    os.makedirs(dataset_dir, exist_ok=True)

    config_path = os.path.join(dataset_dir, "config.json")
    if not os.path.exists(config_path):
        write_json_file(config_path, config)

    for filename, default_value in (
        ("labels.json", {}),
        ("notation_labels.json", {}),
        ("excluded_segments.json", []),
    ):
        path = os.path.join(dataset_dir, filename)
        if not os.path.exists(path):
            write_json_file(path, default_value)

    return config


def ensure_default_datasets(cache_base: Optional[str] = None) -> None:
    for dataset_name in DEFAULT_DATASET_CONFIGS:
        ensure_dataset(dataset_name, cache_base)


def list_library_names(cache_base: Optional[str] = None) -> List[str]:
    libraries_base = get_libraries_base(cache_base)
    if not os.path.isdir(libraries_base):
        return []
    return sorted(
        name
        for name in os.listdir(libraries_base)
        if os.path.isdir(os.path.join(libraries_base, name))
    )


def get_public_libraries(cache_base: Optional[str] = None) -> List[str]:
    discovered = [
        name
        for name in list_library_names(cache_base)
        if name not in {LOCAL_LIBRARY, BACKGROUND_LIBRARY}
    ]
    return discovered or list(DEFAULT_PUBLIC_LIBRARIES)


def get_dataset_libraries(dataset_name: str, cache_base: Optional[str] = None) -> List[str]:
    config = load_dataset_config(dataset_name, cache_base)
    included = config.get("included_libraries", [])
    if included:
        return included
    if dataset_name == "all-public":
        return get_public_libraries(cache_base)
    return []


def get_selected_files(dataset_name: str, cache_base: Optional[str] = None) -> Dict[str, Set[str]]:
    config = load_dataset_config(dataset_name, cache_base)
    selected = config.get("selected_files") or {}
    return {library_name: {str(file_id) for file_id in file_ids} for library_name, file_ids in selected.items()}


def get_selected_files_for_library(
    dataset_name: str,
    library_name: str,
    cache_base: Optional[str] = None,
) -> Optional[Set[str]]:
    selected = get_selected_files(dataset_name, cache_base)
    if library_name not in selected:
        return None
    return selected[library_name]


def is_file_allowed(dataset_name: str, library_name: str, file_id: str, cache_base: Optional[str] = None) -> bool:
    selected = get_selected_files_for_library(dataset_name, library_name, cache_base)
    if selected is None:
        return True
    return str(file_id) in selected


def get_library_catalog_path(library_name: str, cache_base: Optional[str] = None) -> str:
    return os.path.join(get_library_dir(library_name, cache_base), "library.csv")


def read_library_catalog_rows(library_name: str, cache_base: Optional[str] = None) -> List[dict]:
    csv_path = get_library_catalog_path(library_name, cache_base)
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_catalog_rows(
    rows: Sequence[dict],
    selected_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    selected_lookup = {str(value) for value in selected_ids} if selected_ids is not None else None
    filtered = []
    for row in rows:
        file_id = str(row.get("ML Catalog Number", "")).strip()
        if not file_id:
            continue
        if selected_lookup is not None and file_id not in selected_lookup:
            continue
        filtered.append(row)

    if limit is not None:
        return filtered[: max(0, limit)]
    return filtered


def get_dataset_excluded_segments(dataset_name: str, cache_base: Optional[str] = None) -> Set[str]:
    excluded_path = get_dataset_artifact_path(dataset_name, "excluded_segments.json", cache_base=cache_base)
    excluded = read_json_file(excluded_path, [])
    return {str(item) for item in excluded}


def add_dataset_excluded_segment(dataset_name: str, segment_key: str, cache_base: Optional[str] = None) -> None:
    excluded_path = get_dataset_artifact_path(dataset_name, "excluded_segments.json", cache_base=cache_base)
    excluded = list(get_dataset_excluded_segments(dataset_name, cache_base))
    if segment_key not in excluded:
        excluded.append(segment_key)
    write_json_file(excluded_path, sorted(excluded))


@lru_cache(maxsize=64)
def _build_dataset_file_index_cached(
    dataset_name: str,
    relative_dir: str,
    extensions: tuple,
    cache_base: str,
) -> Dict[str, dict]:
    index = {}
    allowed_extensions = set(extensions)
    for library_name in get_dataset_libraries(dataset_name, cache_base):
        selected = get_selected_files_for_library(dataset_name, library_name, cache_base)
        base_dir = os.path.join(get_library_dir(library_name, cache_base), relative_dir)
        if not os.path.isdir(base_dir):
            continue
        for filename in sorted(os.listdir(base_dir)):
            stem, ext = os.path.splitext(filename)
            if ext not in allowed_extensions:
                continue
            if selected is not None and stem not in selected:
                continue
            record = index.setdefault(stem, {"library": library_name, "paths": {}})
            record["paths"][ext] = os.path.join(base_dir, filename)
    return index


def build_dataset_file_index(
    dataset_name: str,
    *,
    relative_dir: str = "audio",
    extensions: Optional[Sequence[str]] = None,
    cache_base: Optional[str] = None,
) -> Dict[str, dict]:
    normalized_cache_base = get_cache_base(cache_base)
    normalized_extensions = tuple(extensions or [".mp3", ".wav", ".npy"])
    return _build_dataset_file_index_cached(
        dataset_name,
        relative_dir,
        normalized_extensions,
        normalized_cache_base,
    )


def resolve_dataset_file_path(
    dataset_name: str,
    file_id: str,
    *,
    relative_dir: str,
    extensions: Optional[Sequence[str]] = None,
    cache_base: Optional[str] = None,
) -> Optional[str]:
    index = build_dataset_file_index(
        dataset_name,
        relative_dir=relative_dir,
        extensions=extensions,
        cache_base=cache_base,
    )
    record = index.get(str(file_id))
    if not record:
        return None
    for ext in extensions or [".mp3", ".wav", ".npy"]:
        if ext in record["paths"]:
            return record["paths"][ext]
    return None


def resolve_dataset_file_library(
    dataset_name: str,
    file_id: str,
    *,
    relative_dir: str = "audio",
    cache_base: Optional[str] = None,
) -> Optional[str]:
    index = build_dataset_file_index(dataset_name, relative_dir=relative_dir, cache_base=cache_base)
    record = index.get(str(file_id))
    return None if record is None else record["library"]


def resolve_dataset_audio_path(
    dataset_name: str,
    file_id: str,
    *,
    denoised: bool = False,
    cache_base: Optional[str] = None,
) -> Optional[str]:
    relative_dir = "audio-denoised" if denoised else "audio"
    extensions = [".wav"] if denoised else [".mp3", ".wav"]
    return resolve_dataset_file_path(
        dataset_name,
        file_id,
        relative_dir=relative_dir,
        extensions=extensions,
        cache_base=cache_base,
    )


def resolve_dataset_embedding_path(
    dataset_name: str,
    file_id: str,
    *,
    denoised: bool = True,
    cache_base: Optional[str] = None,
) -> Optional[str]:
    relative_dir = "embeddings-denoised" if denoised else "embeddings"
    return resolve_dataset_file_path(
        dataset_name,
        file_id,
        relative_dir=relative_dir,
        extensions=[".npy"],
        cache_base=cache_base,
    )


def resolve_dataset_volume_path(
    dataset_name: str,
    file_id: str,
    *,
    denoised: bool = True,
    cache_base: Optional[str] = None,
) -> Optional[str]:
    relative_dir = "embeddings-denoised-volumes" if denoised else "embeddings-volumes"
    return resolve_dataset_file_path(
        dataset_name,
        file_id,
        relative_dir=relative_dir,
        extensions=[".npy"],
        cache_base=cache_base,
    )


def load_dataset_auto_labels(dataset_name: str, cache_base: Optional[str] = None) -> Dict[str, dict]:
    merged = {}
    for library_name in get_dataset_libraries(dataset_name, cache_base):
        labels_path = os.path.join(get_library_dir(library_name, cache_base), "labels", "auto.json")
        labels = read_json_file(labels_path, {})
        selected = get_selected_files_for_library(dataset_name, library_name, cache_base)
        for key, value in labels.items():
            file_id = key.split("-", 1)[0]
            if selected is not None and file_id not in selected:
                continue
            merged[key] = value
    return merged


def _merge_segment_entry(target: Dict[str, List[dict]], file_id: str, segment: dict, library_name: Optional[str]) -> None:
    entry = dict(segment)
    if library_name and "library" not in entry:
        entry["library"] = library_name
    target.setdefault(file_id, []).append(entry)


def load_dataset_segments(dataset_name: str, cache_base: Optional[str] = None) -> Dict[str, List[dict]]:
    excluded = get_dataset_excluded_segments(dataset_name, cache_base)
    dataset_segments_path = get_dataset_artifact_path(dataset_name, "segments.json", cache_base=cache_base)
    merged = {}

    if os.path.exists(dataset_segments_path):
        dataset_segments = read_json_file(dataset_segments_path, {})
        for file_id, segments in dataset_segments.items():
            for segment in segments:
                seg_key = f"{file_id}-{segment['start_time']}-{segment['end_time']}"
                if seg_key in excluded:
                    continue
                _merge_segment_entry(merged, file_id, segment, segment.get("library"))
        return merged

    for library_name in get_dataset_libraries(dataset_name, cache_base):
        selected = get_selected_files_for_library(dataset_name, library_name, cache_base)
        segments_path = os.path.join(get_library_dir(library_name, cache_base), "segments.json")
        library_segments = read_json_file(segments_path, {})
        for file_id, segments in library_segments.items():
            if selected is not None and file_id not in selected:
                continue
            for segment in segments:
                seg_key = f"{file_id}-{segment['start_time']}-{segment['end_time']}"
                if seg_key in excluded:
                    continue
                _merge_segment_entry(merged, file_id, segment, library_name)
    return merged


def find_file_path(
    file_id: str,
    *,
    libraries: Optional[Sequence[str]] = None,
    relative_dir: str = "audio",
    extensions: Optional[Sequence[str]] = None,
    cache_base: Optional[str] = None,
) -> Optional[str]:
    libraries = list(libraries) if libraries is not None else list_library_names(cache_base)
    extensions = list(extensions or [".mp3", ".wav", ".npy"])
    for library_name in libraries:
        base_dir = os.path.join(get_library_dir(library_name, cache_base), relative_dir)
        for extension in extensions:
            candidate = os.path.join(base_dir, f"{file_id}{extension}")
            if os.path.exists(candidate):
                return candidate
    return None


def find_file_library(
    file_id: str,
    *,
    libraries: Optional[Sequence[str]] = None,
    cache_base: Optional[str] = None,
) -> Optional[str]:
    libraries = list(libraries) if libraries is not None else list_library_names(cache_base)
    for library_name in libraries:
        audio_dir = os.path.join(get_library_dir(library_name, cache_base), "audio")
        for extension in (".mp3", ".wav"):
            if os.path.exists(os.path.join(audio_dir, f"{file_id}{extension}")):
                return library_name
    return None
