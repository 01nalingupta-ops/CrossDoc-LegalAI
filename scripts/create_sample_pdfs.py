import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_utils import write_simple_text_pdf

write_simple_text_pdf(
    "sample_docs/sample_master_msa.pdf",
    "Fictional Master Services Agreement",
    [
        "Payment Terms. Customer shall pay all undisputed invoices within Net 30 days.",
        "Liability. Supplier's aggregate liability is capped at fees paid in the prior twelve months.",
        "Confidentiality. Each party shall protect confidential business and technical information.",
        "Termination. Either party may terminate for convenience with 30 days written notice.",
    ],
)
write_simple_text_pdf(
    "sample_docs/sample_service_sow.pdf",
    "Fictional Statement of Work",
    [
        "Payment Terms. Customer shall pay all undisputed invoices within Net 60 days.",
        "Liability. Supplier's liability is uncapped for all claims under this SOW.",
        "Confidentiality. Each party shall protect confidential business and technical information.",
        *[f"Operational Background {i}. The parties coordinate project planning, status meetings, testing, documentation, governance, data handling, and change control." for i in range(1, 9)],
        "Unverified Demo. This sentence triggers an unverified placeholder claim for trust-safety testing.",
    ],
)
