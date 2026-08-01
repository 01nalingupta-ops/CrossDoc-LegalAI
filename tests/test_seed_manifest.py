import json

from seed_manifest import generate_seed_manifest, validate_seed_manifest


def canonical_bytes(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_same_master_seed_produces_byte_identical_manifest():
    first = generate_seed_manifest(master_seed=42, num_pairs=60)
    second = generate_seed_manifest(master_seed=42, num_pairs=60)

    assert canonical_bytes(first) == canonical_bytes(second)


def test_validate_seed_manifest_flags_missing_required_field():
    manifest = generate_seed_manifest(master_seed=42, num_pairs=2)
    del manifest["template_selection_seed"]

    errors = validate_seed_manifest(manifest)

    assert "template_selection_seed is required and must be an integer" in errors
