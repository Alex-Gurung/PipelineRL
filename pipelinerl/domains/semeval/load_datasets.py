import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _read_jsonl(file_path: str) -> list[dict]:
    """Read JSONL file and return list of dicts."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_semeval(
    dataset_names: List[str] | str | None,
    data_dir: str | None = None,
    train_file: str | None = None,
    val_file: str | None = None,
    test_file: str | None = None,
) -> List[dict]:
    """Load SemEval JSONL datasets with minimal processing.

    Expected JSONL format:
    {"messages": [...], "answer": "..."}

    Returns items with added 'dataset' and 'id' fields:
    {"messages": [...], "answer": "...", "dataset": "semeval_train", "id": 0}
    """
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Resolve file paths
    def _resolve(name: str) -> str | None:
        if name == "train":
            return train_file or (Path(data_dir) / "train.jsonl" if data_dir else None)
        if name in ["val", "valid", "validation"]:
            if val_file:
                return val_file
            if data_dir:
                p = Path(data_dir) / "val.jsonl"
                if not p.exists():
                    p = Path(data_dir) / "validation.jsonl"
                return str(p) if p else None
            return None
        if name == "test":
            return test_file or (Path(data_dir) / "test.jsonl" if data_dir else None)
        return None

    out: list[dict] = []
    for name in dataset_names:
        file_path = _resolve(name)
        if not file_path:
            logger.warning(f"No file configured for dataset '{name}'")
            continue

        try:
            items = _read_jsonl(file_path)
        except FileNotFoundError:
            logger.error(f"JSONL not found: {file_path}")
            continue

        dataset_tag = f"semeval_{name}"
        # Add dataset name and ID to each item
        for i, item in enumerate(items):
            item["dataset"] = dataset_tag
            item["id"] = i

        logger.info(f"Loaded {len(items)} samples from {file_path} as {dataset_tag}")
        out.extend(items)

    return out