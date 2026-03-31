import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowtools.datasets import (
    DEFAULT_PUBLIC_LIBRARIES,
    ensure_default_datasets,
    find_file_library,
    find_file_path,
    get_dataset_libraries,
    get_dataset_import_path,
    resolve_dataset_audio_path,
    resolve_dataset_embedding_path,
    get_selected_files,
)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ML Catalog Number", "Recordist", "Media notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_fake_cache(tmp_path):
    cache_dir = tmp_path / ".cache"
    rows = [
        {"ML Catalog Number": "111", "Recordist": "A", "Media notes": "m1"},
        {"ML Catalog Number": "222", "Recordist": "B", "Media notes": "m2"},
    ]
    for library_name in ["macaulay", "xeno-canto", "backgrounds"]:
        library_dir = cache_dir / "libraries" / library_name
        (library_dir / "audio").mkdir(parents=True, exist_ok=True)
        if library_name != "backgrounds":
            write_csv(str(library_dir / "library.csv"), rows)
    (cache_dir / "libraries" / "macaulay" / "audio" / "111.mp3").write_bytes(b"m")
    (cache_dir / "libraries" / "xeno-canto" / "audio" / "222.mp3").write_bytes(b"x")
    return cache_dir


def test_default_datasets_are_created_in_temp_cache(tmp_path):
    cache_dir = build_fake_cache(tmp_path)

    ensure_default_datasets(str(cache_dir))

    assert (cache_dir / "datasets" / "starter" / "config.json").exists()
    assert (cache_dir / "datasets" / "all-public" / "config.json").exists()


def test_all_public_excludes_background_library(tmp_path):
    cache_dir = build_fake_cache(tmp_path)

    ensure_default_datasets(str(cache_dir))
    libraries = get_dataset_libraries("all-public", str(cache_dir))

    assert libraries == DEFAULT_PUBLIC_LIBRARIES
    assert "backgrounds" not in libraries


def test_starter_defaults_to_disk_config_for_selected_files(tmp_path):
    cache_dir = build_fake_cache(tmp_path)

    ensure_default_datasets(str(cache_dir))
    starter_config_path = cache_dir / "datasets" / "starter" / "config.json"
    starter_config_path.write_text(
        '{"name":"starter","included_libraries":["macaulay","xeno-canto"],'
        '"selected_files":{"macaulay":["111"],"xeno-canto":["222"]}}',
        encoding="utf-8",
    )
    selected = get_selected_files("starter", str(cache_dir))

    assert selected["macaulay"] == {"111"}
    assert selected["xeno-canto"] == {"222"}


def test_find_file_helpers_respect_allowed_libraries(tmp_path):
    cache_dir = build_fake_cache(tmp_path)

    audio_path = find_file_path("111", libraries=["macaulay"], cache_base=str(cache_dir))
    assert str(audio_path).endswith("macaulay/audio/111.mp3")

    library = find_file_library("222", libraries=["xeno-canto"], cache_base=str(cache_dir))
    assert library == "xeno-canto"


def test_dataset_imports_resolve_before_library_files(tmp_path):
    cache_dir = build_fake_cache(tmp_path)

    ensure_default_datasets(str(cache_dir))
    dataset_audio_path = Path(get_dataset_import_path("starter", "audio", "111.wav", cache_base=str(cache_dir)))
    dataset_audio_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_audio_path.write_bytes(b"dataset")

    dataset_embedding_path = Path(get_dataset_import_path("starter", "embeddings", "111.npy", cache_base=str(cache_dir)))
    dataset_embedding_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_embedding_path.write_bytes(b"embedding")

    resolved_audio = resolve_dataset_audio_path("starter", "111", cache_base=str(cache_dir))
    resolved_embedding = resolve_dataset_embedding_path("starter", "111", denoised=False, cache_base=str(cache_dir))

    assert resolved_audio == str(dataset_audio_path)
    assert resolved_embedding == str(dataset_embedding_path)
