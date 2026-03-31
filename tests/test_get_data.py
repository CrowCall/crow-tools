import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "get-data.py"


def load_get_data_module():
    spec = importlib.util.spec_from_file_location("get_data_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dry_run_skips_pipeline(monkeypatch, tmp_path):
    get_data = load_get_data_module()
    calls = []

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(get_data, "run_pipeline", fake_pipeline)

    rc = get_data.main(["--dataset", "starter", "--cache-dir", str(tmp_path), "--dry-run"])

    assert rc == 0
    assert calls == []


def test_starter_pipeline_uses_deterministic_selected_files(monkeypatch, tmp_path):
    get_data = load_get_data_module()
    recorded = []
    selected_files = {
        "macaulay": {"m1", "m2"},
        "xeno-canto": {"x1", "x2"},
    }

    def record(name):
        def inner(*args, **kwargs):
            recorded.append((name, args, kwargs))
        return inner

    monkeypatch.setattr(
        get_data,
        "resolve_pipeline_functions",
        lambda: {
            "download_backgrounds": record("download_backgrounds"),
            "download_macaulay": record("download_macaulay"),
            "download_xeno": record("download_xeno"),
            "start_denoising": record("start_denoising"),
            "start_detections": record("start_detections"),
            "start_embeddings": record("start_embeddings"),
        },
    )
    monkeypatch.setattr(get_data, "get_selected_files", lambda dataset_name, cache_dir: selected_files)

    rc = get_data.main(["--dataset", "starter", "--cache-dir", str(tmp_path), "--include-backgrounds"])

    assert rc == 0

    macaulay_call = next(item for item in recorded if item[0] == "download_macaulay")
    xeno_call = next(item for item in recorded if item[0] == "download_xeno")
    denoise_call = next(item for item in recorded if item[0] == "start_denoising")
    detect_call = next(item for item in recorded if item[0] == "start_detections")
    embed_calls = [item for item in recorded if item[0] == "start_embeddings"]

    assert macaulay_call[2]["selected_ids"] == selected_files["macaulay"]
    assert xeno_call[2]["selected_ids"] == selected_files["xeno-canto"]
    assert denoise_call[2]["libraries"] == ["macaulay", "xeno-canto"]
    assert denoise_call[2]["selected_ids_by_library"]["macaulay"] == selected_files["macaulay"]
    assert detect_call[2]["libraries"] == ["macaulay", "xeno-canto"]
    assert [call[2]["denoised"] for call in embed_calls] == [False, True]
