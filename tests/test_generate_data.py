import json
import shutil
from pathlib import Path

from generate_data import generate_dataset, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "base_corpus"


def canonical_bytes(data):
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def test_same_master_seed_produces_byte_identical_dataset_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first, first_manifest = generate_dataset(CORPUS, num_pairs=60, master_seed=42)
    shutil.rmtree(tmp_path / "generated_contracts")
    second, second_manifest = generate_dataset(CORPUS, num_pairs=60, master_seed=42)

    assert canonical_bytes(first) == canonical_bytes(second)
    assert canonical_bytes(first_manifest) == canonical_bytes(second_manifest)


def test_clean_pair_ratio_lands_in_target_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entries, _ = generate_dataset(CORPUS, num_pairs=100, master_seed=99)

    clean_count = sum(1 for entry in entries if not entry["has_contradiction"])
    ratio = clean_count / len(entries)

    assert 0.15 <= ratio <= 0.20


def test_every_generated_entry_passes_substring_self_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entries, _ = generate_dataset(CORPUS, num_pairs=60, master_seed=7)

    assert validate_dataset(entries) == []
