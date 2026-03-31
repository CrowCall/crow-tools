import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_module(path, name, stub_modules=None):
    stub_modules = stub_modules or {}
    previous = {}
    for module_name, module in stub_modules.items():
        previous[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module

    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module


def test_detect_all_main_resolves_dataset_selection(monkeypatch, tmp_path):
    detector_pkg = types.ModuleType("detector")
    detector_detect = types.ModuleType("detector.detect")
    detector_detect.detect_file_segments = lambda *args, **kwargs: ([], None, None)
    detector_pkg.detect = detector_detect

    classifier_pkg = types.ModuleType("classifier")
    classifier_classify = types.ModuleType("classifier.classify")
    classifier_classify.predict_embedding = lambda *args, **kwargs: {}
    classifier_pkg.classify = classifier_classify

    module = load_module(
        ROOT / "detector" / "detect_all.py",
        "test_detect_all_module",
        {
            "detector": detector_pkg,
            "detector.detect": detector_detect,
            "classifier": classifier_pkg,
            "classifier.classify": classifier_classify,
        },
    )

    recorded = {}
    monkeypatch.setattr(module, "get_dataset_libraries", lambda dataset, cache: ["macaulay", "xeno-canto"])
    monkeypatch.setattr(
        module,
        "get_selected_files_for_library",
        lambda dataset, library, cache: {f"{library}-1"},
    )
    monkeypatch.setattr(
        module,
        "start_detections",
        lambda **kwargs: recorded.update(kwargs),
    )

    module.main(["--dataset", "starter", "--cache-dir", str(tmp_path)])

    assert recorded["libraries"] == ["macaulay", "xeno-canto"]
    assert recorded["selected_ids_by_library"] == {
        "macaulay": {"macaulay-1"},
        "xeno-canto": {"xeno-canto-1"},
    }
    assert recorded["cache_base"] == str(tmp_path)


def test_embed_all_main_runs_raw_and_denoised(monkeypatch, tmp_path):
    embedder_pkg = types.ModuleType("embedder")
    embedder_embed = types.ModuleType("embedder.embed")
    embedder_embed.generate_embeddings = lambda *args, **kwargs: []
    embedder_ispa = types.ModuleType("embedder.ispa")
    embedder_ispa.utils = types.SimpleNamespace()
    embedder_pkg.embed = embedder_embed
    embedder_pkg.ispa = embedder_ispa

    module = load_module(
        ROOT / "embedder" / "embed_all.py",
        "test_embed_all_module",
        {
            "embedder": embedder_pkg,
            "embedder.embed": embedder_embed,
            "embedder.ispa": embedder_ispa,
        },
    )

    calls = []
    monkeypatch.setattr(module, "get_dataset_libraries", lambda dataset, cache: ["macaulay"])
    monkeypatch.setattr(module, "get_selected_files", lambda dataset, cache: {"macaulay": {"105346"}})
    monkeypatch.setattr(module, "start_embeddings", lambda **kwargs: calls.append(kwargs))

    module.main(["--dataset", "starter", "--cache-dir", str(tmp_path)])

    assert [call["denoised"] for call in calls] == [False, True]
    assert all(call["libraries"] == ["macaulay"] for call in calls)
    assert all(call["selected_ids_by_library"] == {"macaulay": {"105346"}} for call in calls)
    assert all(call["cache_base"] == str(tmp_path) for call in calls)


def test_denoise_all_main_resolves_dataset_selection(monkeypatch, tmp_path):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda value: value

    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.save = lambda *args, **kwargs: None

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.load = lambda *args, **kwargs: ([], 16000)

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    fake_pretrained = types.ModuleType("biodenoising.pretrained")

    class FakeModel:
        sample_rate = 16000
        chin = 1

        def to(self, device):
            return self

    fake_pretrained.biodenoising16k_dns48 = lambda: FakeModel()

    fake_dsp = types.ModuleType("biodenoising.denoiser.dsp")
    fake_dsp.convert_audio = lambda wav, sr, sample_rate, chin: wav

    biodenoising_pkg = types.ModuleType("biodenoising")
    biodenoising_pkg.pretrained = fake_pretrained

    biodenoising_denoiser_pkg = types.ModuleType("biodenoising.denoiser")
    biodenoising_denoiser_pkg.dsp = fake_dsp

    module = load_module(
        ROOT / "denoiser" / "denoise_all.py",
        "test_denoise_all_module",
        {
            "torch": fake_torch,
            "torchaudio": fake_torchaudio,
            "librosa": fake_librosa,
            "tqdm": fake_tqdm,
            "biodenoising": biodenoising_pkg,
            "biodenoising.pretrained": fake_pretrained,
            "biodenoising.denoiser": biodenoising_denoiser_pkg,
            "biodenoising.denoiser.dsp": fake_dsp,
        },
    )

    recorded = {}
    monkeypatch.setattr(module, "get_dataset_libraries", lambda dataset, cache: ["macaulay"])
    monkeypatch.setattr(module, "get_selected_files", lambda dataset, cache: {"macaulay": {"105346"}})
    monkeypatch.setattr(module, "start_denoising", lambda **kwargs: recorded.update(kwargs))

    module.main(["--dataset", "starter", "--cache-dir", str(tmp_path)])

    assert recorded["libraries"] == ["macaulay"]
    assert recorded["selected_ids_by_library"] == {"macaulay": {"105346"}}
    assert recorded["cache_base"] == str(tmp_path)


def test_review_detections_cli_accepts_non_interactive_args(monkeypatch, tmp_path):
    module = load_module(
        ROOT / "classifier" / "review_detections.py",
        "test_review_detections_module",
    )

    recorded = {}
    monkeypatch.setattr(
        module,
        "main",
        lambda attribute, cluster, offset, dataset_name="all-public", cache_base=None: recorded.update(
            {
                "attribute": attribute,
                "cluster": cluster,
                "offset": offset,
                "dataset_name": dataset_name,
                "cache_base": cache_base,
            }
        ),
    )

    module.main_cli(
        [
            "--dataset",
            "starter",
            "--cache-dir",
            str(tmp_path),
            "--attribute",
            "quality:2",
            "--cluster",
            "7",
            "--offset",
            "3",
        ]
    )

    assert recorded == {
        "attribute": "quality:2",
        "cluster": 7,
        "offset": 3,
        "dataset_name": "starter",
        "cache_base": str(tmp_path),
    }
