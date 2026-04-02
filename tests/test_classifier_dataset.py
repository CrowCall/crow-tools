import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.dataset import CrowDataset, split_train_val_dataset


def test_split_train_val_dataset_is_deterministic(tmp_path):
    cache_dir = tmp_path / ".cache"
    dataset_dir = cache_dir / "datasets" / "starter"
    embeddings_dir = dataset_dir / "imports" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    for index in range(10):
        file_id = f"file{index}"
        np.save(embeddings_dir / f"{file_id}.npy", np.zeros((2, 768), dtype=np.float32))
        labels[f"{file_id}-0-1"] = {
            "reviewed": True,
            "crowCount": 1,
            "crowAge": 1,
            "quality": 2,
        }

    (dataset_dir / "config.json").write_text(
        json.dumps({"name": "starter", "included_libraries": []}),
        encoding="utf-8",
    )
    (dataset_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    dataset = CrowDataset(dataset_name="starter", cache_base=str(cache_dir))
    train_a, val_a = split_train_val_dataset(dataset)
    train_b, val_b = split_train_val_dataset(dataset)

    assert dataset.keys == sorted(dataset.keys)
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
