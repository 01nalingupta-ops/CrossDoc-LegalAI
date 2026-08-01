"""Generate and validate CrossDoc-LegalAI seed manifests.

Deterministic sub-seed derivation scheme
========================================
Given a non-negative integer ``master_seed`` and a UTF-8 label, compute:

1. ``message = f"CrossDoc-LegalAI-seed-v1:{master_seed}:{label}"`` exactly, with no
   whitespace before/after the colons.
2. ``digest = SHA256(message encoded as UTF-8)``.
3. Interpret the first 8 digest bytes as an unsigned big-endian integer.
4. Return that integer modulo ``2_147_483_647`` (the largest signed 32-bit prime).

Dataset-generation per-pair seeds use labels ``dataset_generation_seed:0000`` through
``dataset_generation_seed:{num_pairs-1:04d}``, in order. All other manifest seeds use the
field name as the label, for example ``template_selection_seed``.

This scheme is intentionally language-neutral so other modules can reimplement it and
obtain byte-identical seed manifests from the same ``master_seed`` and ``num_pairs``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

SEED_MODULUS = 2_147_483_647
DERIVATION_PREFIX = "CrossDoc-LegalAI-seed-v1"
DATASET_HASH_PLACEHOLDER = "0" * 64
MODEL_VERSION_SNAPSHOT = {
    "generator_model": "placeholder-generator-model",
    "candidate_1": "placeholder-candidate-1",
    "candidate_2": "placeholder-candidate-2",
    "candidate_3": "placeholder-candidate-3",
    "candidate_4": "placeholder-candidate-4",
}
REQUIRED_SEED_FIELDS = [
    "template_selection_seed",
    "defect_category_seed",
    "defect_location_seed",
    "severity_assignment_seed",
    "clean_pair_selection_seed",
    "retrieval_tiebreak_seed",
    "bootstrap_resampling_seed",
]
REQUIRED_MODEL_FIELDS = ["generator_model", "candidate_1", "candidate_2", "candidate_3", "candidate_4"]


def derive_subseed(master_seed: int, label: str) -> int:
    """Derive one deterministic 31-bit positive sub-seed from ``master_seed`` and ``label``."""
    if not isinstance(master_seed, int) or isinstance(master_seed, bool) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    message = f"{DERIVATION_PREFIX}:{master_seed}:{label}".encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % SEED_MODULUS


def generate_seed_manifest(master_seed: int, num_pairs: int) -> dict[str, Any]:
    """Return a deterministic seed manifest matching the published JSON contract."""
    if not isinstance(num_pairs, int) or isinstance(num_pairs, bool) or num_pairs < 0:
        raise ValueError("num_pairs must be a non-negative integer")
    manifest: dict[str, Any] = {
        "master_seed": master_seed,
        "dataset_generation_seeds": [
            derive_subseed(master_seed, f"dataset_generation_seed:{i:04d}") for i in range(num_pairs)
        ],
        "dataset_content_hash": DATASET_HASH_PLACEHOLDER,
        "model_version_snapshot": dict(MODEL_VERSION_SNAPSHOT),
    }
    for field in REQUIRED_SEED_FIELDS:
        manifest[field] = derive_subseed(master_seed, field)
    # Emit fields in the contract's documented order.
    return {
        "master_seed": manifest["master_seed"],
        "dataset_generation_seeds": manifest["dataset_generation_seeds"],
        "template_selection_seed": manifest["template_selection_seed"],
        "defect_category_seed": manifest["defect_category_seed"],
        "defect_location_seed": manifest["defect_location_seed"],
        "severity_assignment_seed": manifest["severity_assignment_seed"],
        "clean_pair_selection_seed": manifest["clean_pair_selection_seed"],
        "retrieval_tiebreak_seed": manifest["retrieval_tiebreak_seed"],
        "bootstrap_resampling_seed": manifest["bootstrap_resampling_seed"],
        "dataset_content_hash": manifest["dataset_content_hash"],
        "model_version_snapshot": manifest["model_version_snapshot"],
    }


def validate_seed_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return validation errors for a seed manifest, or an empty list when valid."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    if not _is_int(manifest.get("master_seed")):
        errors.append("master_seed is required and must be an integer")

    seeds = manifest.get("dataset_generation_seeds")
    if not isinstance(seeds, list) or not all(_is_int(seed) for seed in seeds):
        errors.append("dataset_generation_seeds is required and must be a list of integers")

    for field in REQUIRED_SEED_FIELDS:
        if not _is_int(manifest.get(field)):
            errors.append(f"{field} is required and must be an integer")

    dataset_hash = manifest.get("dataset_content_hash")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64 or not _is_hex(dataset_hash):
        errors.append("dataset_content_hash is required and must be a 64-character SHA-256 hex string")

    snapshot = manifest.get("model_version_snapshot")
    if not isinstance(snapshot, dict):
        errors.append("model_version_snapshot is required and must be an object")
    else:
        for field in REQUIRED_MODEL_FIELDS:
            if not isinstance(snapshot.get(field), str) or not snapshot.get(field):
                errors.append(f"model_version_snapshot.{field} is required and must be a non-empty string")

    return errors


def write_manifest(manifest: dict[str, Any], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value: str) -> bool:
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic CrossDoc-LegalAI seed manifest.")
    parser.add_argument("--master-seed", required=True, type=int)
    parser.add_argument("--num-pairs", required=True, type=int)
    parser.add_argument("--out", required=True, help="Output path for seed_manifest.json")
    args = parser.parse_args(argv)

    manifest = generate_seed_manifest(args.master_seed, args.num_pairs)
    errors = validate_seed_manifest(manifest)
    if errors:
        raise SystemExit("Invalid generated manifest:\n" + "\n".join(errors))
    write_manifest(manifest, args.out)
    print(f"Wrote seed manifest to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
