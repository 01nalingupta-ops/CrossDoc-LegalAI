import pytest

from clause_classifier import CATEGORY_LABELS, OTHER_CATEGORY, classify_clause, infer_category
from generate_data import DEFECT_CATEGORIES


@pytest.mark.parametrize(
    ("category", "text"),
    [
        ("Payment Terms", "Payment Terms: Client must pay all undisputed invoices within Net 30 days."),
        ("Liability Cap", "Liability Cap: Supplier liability is capped at fees paid in the prior year."),
        ("IP Ownership", "IP Ownership: Client owns custom deliverables and work product."),
        ("Termination Notice", "Termination Notice: Either party may terminate on written notice."),
        ("Governing Jurisdiction", "Governing Jurisdiction: New York law and venue govern disputes."),
        ("Confidentiality Scope", "Confidentiality Scope: Confidential information and trade secrets are protected."),
        ("Indemnification", "Indemnification: Vendor will indemnify Client for third party claims."),
        ("Insurance Requirements", "Insurance Requirements: Vendor must maintain insurance coverage certificates."),
    ],
)
def test_classifier_covers_all_fixed_generate_data_categories(category, text):
    result = classify_clause(text)

    assert DEFECT_CATEGORIES == CATEGORY_LABELS[:-1]
    assert result["category"] == category
    assert result["confidence"] > 0.0
    assert infer_category(text) == category


def test_classifier_returns_other_for_unrelated_clause():
    result = classify_clause("The project kickoff meeting will be scheduled by email.")

    assert result["category"] == OTHER_CATEGORY
    assert result["confidence"] == 0.0
