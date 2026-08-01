"""Synthetic adversarial dataset generator for CrossDoc-LegalAI Part 3.

Seed derivation scheme
======================
All deterministic choices derive from ``master_seed`` using SHA-256:
``CrossDoc-LegalAI-dataset-v1:<master_seed>:<purpose>:<pair_index>``. The first 8 digest
bytes are interpreted as an unsigned big-endian integer and reduced modulo
``2_147_483_647``. Global manifest seeds omit ``pair_index`` and use
``CrossDoc-LegalAI-dataset-v1:<master_seed>:<purpose>``.

Output schema
=============
``benchmark_dataset.json`` is a JSON array. Every entry has exactly these keys:
``pair_id``, ``seed``, ``master_doc_path``, ``service_doc_path``, ``has_contradiction``,
``contradiction_category``, ``severity``, ``master_clause_location``,
``service_clause_location``, ``master_exact_text``, ``service_exact_text``,
``generation_model``, and ``generation_timestamp``.

This module uses deterministic local template rewriting only. It performs no LLM calls,
retrieval, guardrail logic, or evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from random import Random
from typing import Any, Sequence

DERIVATION_PREFIX = "CrossDoc-LegalAI-dataset-v1"
SEED_MODULUS = 2_147_483_647
GENERATION_MODEL = "deterministic-template-rewriter-v1"
CLEAN_RATIO = 0.17
NEAR_DUPLICATE_THRESHOLD = 0.999

DEFECT_CATEGORIES = [
    "Payment Terms",
    "Liability Cap",
    "IP Ownership",
    "Termination Notice",
    "Governing Jurisdiction",
    "Confidentiality Scope",
    "Indemnification",
    "Insurance Requirements",
]
SEVERITIES = ["High", "Medium", "Low"]
CLAUSE_IDS = {
    "Payment Terms": "payment_terms",
    "Liability Cap": "liability_cap",
    "IP Ownership": "ip_ownership",
    "Termination Notice": "termination_notice",
    "Governing Jurisdiction": "governing_jurisdiction",
    "Confidentiality Scope": "confidentiality_scope",
    "Indemnification": "indemnification",
    "Insurance Requirements": "insurance_requirements",
}

REWRITES = {
    "Payment Terms": {
        "High": "Payment Terms: Client must pay all undisputed invoices within ten (10) days after receipt, and any longer payment period in the master agreement is superseded for this SOW.",
        "Medium": "Payment Terms: Client must pay all undisputed invoices within fifteen (15) days after receipt for this SOW.",
        "Low": "Payment Terms: Client should target payment within twenty (20) days after receipt unless otherwise approved in writing.",
    },
    "Liability Cap": {
        "High": "Liability Cap: Vendor's liability for this SOW is unlimited, notwithstanding any cap stated in the master agreement.",
        "Medium": "Liability Cap: Vendor's liability for this SOW is capped at three times the fees paid under this SOW.",
        "Low": "Liability Cap: Vendor's liability for this SOW is capped at two times the monthly fees, regardless of the master agreement.",
    },
    "IP Ownership": {
        "High": "IP Ownership: Vendor retains all ownership rights in deliverables created under this SOW, including custom work product paid for by Client.",
        "Medium": "IP Ownership: Vendor owns reusable deliverables and grants Client only an internal-use license for custom materials.",
        "Low": "IP Ownership: Vendor may reuse and sublicense project deliverables unless Client objects within five days.",
    },
    "Termination Notice": {
        "High": "Termination Notice: Either party may terminate this SOW for convenience upon five (5) days' written notice.",
        "Medium": "Termination Notice: Either party may terminate this SOW for convenience upon fifteen (15) days' written notice.",
        "Low": "Termination Notice: Either party may terminate this SOW for convenience upon forty-five (45) days' written notice.",
    },
    "Governing Jurisdiction": {
        "High": "Governing Jurisdiction: This SOW is governed by California law and exclusive venue is in San Francisco County, California.",
        "Medium": "Governing Jurisdiction: This SOW is governed by New York law for disputes relating only to services.",
        "Low": "Governing Jurisdiction: Operational disputes under this SOW may be brought in the courts of Austin, Texas.",
    },
    "Confidentiality Scope": {
        "High": "Confidentiality Scope: Only Client's information is confidential, and Vendor's confidentiality obligations expire after six (6) months.",
        "Medium": "Confidentiality Scope: Vendor's confidentiality obligations survive for one (1) year after disclosure under this SOW.",
        "Low": "Confidentiality Scope: Confidentiality obligations apply only to materials marked confidential at the time of disclosure.",
    },
    "Indemnification": {
        "High": "Indemnification: Client alone will indemnify Vendor for all third-party claims arising from this SOW.",
        "Medium": "Indemnification: Vendor has no indemnity obligation for intellectual property claims arising from approved deliverables.",
        "Low": "Indemnification: Vendor indemnity applies only after Client first exhausts available insurance proceeds.",
    },
    "Insurance Requirements": {
        "High": "Insurance Requirements: Vendor is not required to maintain insurance coverage for work performed under this SOW.",
        "Medium": "Insurance Requirements: Vendor must maintain only $250,000 in commercial general liability coverage for this SOW.",
        "Low": "Insurance Requirements: Vendor may satisfy insurance requirements using self-insurance without certificates of coverage.",
    },
}


@dataclass(frozen=True)
class TemplatePair:
    name: str
    master_path: Path
    service_path: Path
    master_text: str
    service_text: str


def derive_seed(master_seed: int, purpose: str, pair_index: int | None = None) -> int:
    if pair_index is None:
        message = f"{DERIVATION_PREFIX}:{master_seed}:{purpose}"
    else:
        message = f"{DERIVATION_PREFIX}:{master_seed}:{purpose}:{pair_index}"
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % SEED_MODULUS


def generate_dataset(corpus_dir: str | Path, num_pairs: int = 60, master_seed: int = 42, out: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    corpus = load_template_pairs(Path(corpus_dir))
    generated_dir = (Path(out).resolve().parent if out else Path.cwd()) / "generated_contracts"
    generated_dir.mkdir(parents=True, exist_ok=True)

    clean_indices = choose_clean_indices(master_seed, num_pairs)
    entries: list[dict[str, Any]] = []
    seen_texts: list[str] = []
    attempt = 0
    while len(entries) < num_pairs and attempt < num_pairs * 20:
        pair_index = len(entries)
        candidate = build_entry(pair_index, attempt, master_seed, corpus, clean_indices, generated_dir)
        combined_text = candidate.pop("_combined_text")
        if is_near_duplicate(combined_text, seen_texts):
            attempt += 1
            continue
        validate_entry_substrings(candidate)
        seen_texts.append(combined_text)
        entries.append(candidate)
        attempt += 1
    if len(entries) != num_pairs:
        raise RuntimeError(f"could not generate {num_pairs} unique pairs after {attempt} attempts")

    manifest = build_seed_manifest(master_seed, num_pairs)
    if out:
        out_path = Path(out)
        write_json(entries, out_path)
        manifest["dataset_content_hash"] = sha256_file(out_path)
        write_json(manifest, out_path.with_name("dataset_generation_seeds.json"))
    else:
        serialized = canonical_json(entries)
        manifest["dataset_content_hash"] = hashlib.sha256(serialized).hexdigest()
    return entries, manifest


def build_entry(pair_index: int, attempt: int, master_seed: int, corpus: Sequence[TemplatePair], clean_indices: set[int], generated_dir: Path) -> dict[str, Any]:
    pair_seed = derive_seed(master_seed, "dataset_generation_seed", pair_index)
    template_rng = Random(derive_seed(master_seed, "template_selection", attempt))
    category_rng = Random(derive_seed(master_seed, "defect_category", attempt))
    severity_rng = Random(derive_seed(master_seed, "severity_assignment", attempt))
    template = corpus[template_rng.randrange(len(corpus))]

    clean = pair_index in clean_indices
    if clean:
        category = None
        severity = "None"
        clause_id = CLAUSE_IDS[DEFECT_CATEGORIES[attempt % len(DEFECT_CATEGORIES)]]
        service_text = template.service_text
    else:
        category = DEFECT_CATEGORIES[category_rng.randrange(len(DEFECT_CATEGORIES))]
        severity = SEVERITIES[severity_rng.randrange(len(SEVERITIES))]
        clause_id = CLAUSE_IDS[category]
        service_text = rewrite_service_clause(template.service_text, clause_id, REWRITES[category][severity])

    pair_id = f"pair-{pair_index + 1:04d}"
    service_text = service_text.rstrip() + f"\n[administrative_note] Synthetic benchmark reference {pair_id} generated for testing only.\n"
    master_out = generated_dir / f"{pair_id}_master.txt"
    service_out = generated_dir / f"{pair_id}_service.txt"
    master_out.write_text(template.master_text, encoding="utf-8")
    service_out.write_text(service_text, encoding="utf-8")

    master_clause_id = clause_id
    master_exact_text, master_offset = extract_clause(template.master_text, master_clause_id)
    service_exact_text, service_offset = extract_clause(service_text, clause_id)
    return {
        "pair_id": pair_id,
        "seed": pair_seed,
        "master_doc_path": str(master_out),
        "service_doc_path": str(service_out),
        "has_contradiction": not clean,
        "contradiction_category": category,
        "severity": severity,
        "master_clause_location": {"page": 1, "char_offset": master_offset, "clause_id": master_clause_id},
        "service_clause_location": {"page": 1, "char_offset": service_offset, "clause_id": clause_id},
        "master_exact_text": master_exact_text,
        "service_exact_text": service_exact_text,
        "generation_model": GENERATION_MODEL,
        "generation_timestamp": deterministic_timestamp(master_seed, pair_index),
        "_combined_text": template.master_text + "\n" + service_text,
    }


def choose_clean_indices(master_seed: int, num_pairs: int) -> set[int]:
    clean_count = max(1, round(num_pairs * CLEAN_RATIO))
    scores = [(derive_seed(master_seed, "clean_pair_selection", i), i) for i in range(num_pairs)]
    return {i for _, i in sorted(scores)[:clean_count]}


def rewrite_service_clause(service_text: str, clause_id: str, replacement: str) -> str:
    pattern = re.compile(rf"^\[{re.escape(clause_id)}\]\s.*$", flags=re.MULTILINE)
    rewritten, count = pattern.subn(f"[{clause_id}] {replacement}", service_text, count=1)
    if count != 1:
        raise ValueError(f"service template missing clause {clause_id}")
    return rewritten


def extract_clause(text: str, clause_id: str) -> tuple[str, int]:
    match = re.search(rf"^\[{re.escape(clause_id)}\]\s.*$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"document missing clause {clause_id}")
    return match.group(0), match.start()


def validate_entry_substrings(entry: dict[str, Any], tolerance: int = 5) -> None:
    master_text = Path(entry["master_doc_path"]).read_text(encoding="utf-8")
    service_text = Path(entry["service_doc_path"]).read_text(encoding="utf-8")
    _validate_quote(master_text, entry["master_exact_text"], entry["master_clause_location"]["char_offset"], tolerance)
    _validate_quote(service_text, entry["service_exact_text"], entry["service_clause_location"]["char_offset"], tolerance)


def validate_dataset(entries: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        try:
            validate_entry_substrings(entry)
        except Exception as exc:  # validation should report all bad entries to callers/tests
            errors.append(f"{entry.get('pair_id', '<unknown>')}: {exc}")
    return errors


def _validate_quote(document_text: str, quote: str, offset: int, tolerance: int) -> None:
    if quote not in document_text:
        raise ValueError("quote is not a substring of document")
    start = document_text.find(quote)
    if abs(start - offset) > tolerance:
        raise ValueError(f"quote offset {offset} is not near actual offset {start}")


def is_near_duplicate(text: str, seen_texts: Sequence[str]) -> bool:
    return any(token_jaccard(text, seen) >= NEAR_DUPLICATE_THRESHOLD for seen in seen_texts)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_template_pairs(corpus_dir: Path) -> list[TemplatePair]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    pairs: list[TemplatePair] = []
    for master_path in sorted(corpus_dir.glob("*_msa.txt")):
        name = master_path.name[: -len("_msa.txt")]
        service_path = corpus_dir / f"{name}_sow.txt"
        if service_path.exists():
            pairs.append(TemplatePair(name, master_path, service_path, master_path.read_text(encoding="utf-8"), service_path.read_text(encoding="utf-8")))
    if not pairs:
        raise ValueError(f"no template pairs found in {corpus_dir}; expected *_msa.txt and matching *_sow.txt")
    return pairs


def build_seed_manifest(master_seed: int, num_pairs: int) -> dict[str, Any]:
    return {
        "master_seed": master_seed,
        "dataset_generation_seeds": [derive_seed(master_seed, "dataset_generation_seed", i) for i in range(num_pairs)],
        "template_selection_seed": derive_seed(master_seed, "template_selection"),
        "defect_category_seed": derive_seed(master_seed, "defect_category"),
        "defect_location_seed": derive_seed(master_seed, "defect_location"),
        "severity_assignment_seed": derive_seed(master_seed, "severity_assignment"),
        "clean_pair_selection_seed": derive_seed(master_seed, "clean_pair_selection"),
        "dataset_content_hash": "0" * 64,
        "model_version_snapshot": {"generator_model": GENERATION_MODEL},
    }


def deterministic_timestamp(master_seed: int, pair_index: int) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seconds = derive_seed(master_seed, "generation_timestamp", pair_index) % (365 * 24 * 60 * 60)
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(data))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic legal contradiction benchmark dataset.")
    parser.add_argument("--corpus-dir", required=True, help="Directory containing *_msa.txt/*_sow.txt template pairs")
    parser.add_argument("--num-pairs", type=int, default=60, help="Number of benchmark pairs to generate")
    parser.add_argument("--master-seed", type=int, default=42, help="Top-level deterministic seed")
    parser.add_argument("--out", required=True, help="Output benchmark_dataset.json path")
    args = parser.parse_args(argv)

    entries, manifest = generate_dataset(args.corpus_dir, args.num_pairs, args.master_seed, args.out)
    errors = validate_dataset(entries)
    if errors:
        raise SystemExit("Generated dataset failed validation:\n" + "\n".join(errors))
    print(f"Wrote {len(entries)} benchmark pairs to {args.out}")
    print(f"Wrote seed manifest fragment to {Path(args.out).with_name('dataset_generation_seeds.json')}")
    print(f"dataset_content_hash={manifest['dataset_content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
