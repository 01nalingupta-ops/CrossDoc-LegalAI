"""Streamlit UI for CrossDoc-LegalAI Part 7.

Swap the stub pipeline for real modules by changing this import line only, provided the
replacement exports parse_document, retrieve_matches, and run_auditor_and_guardrail with
the same signatures documented in INTEGRATION.md.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC

from pdf_utils import build_report_pdf
from pipeline_stubs import infer_category, parse_document, retrieve_matches, run_auditor_and_guardrail, sort_predictions_for_display


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="CrossDoc-LegalAI", layout="wide")
    st.title("CrossDoc-LegalAI: Master vs. Service Document")
    st.caption("Standalone Part 7 demo using offline stubs. No API keys required.")

    left, right = st.columns(2)
    with left:
        master_file = st.file_uploader("Upload Master Agreement PDF", type=["pdf"], key="master")
    with right:
        service_file = st.file_uploader("Upload Service / SOW PDF", type=["pdf"], key="service")

    if not master_file or not service_file:
        st.info("Upload both documents, or use the sample PDFs in sample_docs/ to run the demo.")
        return

    if st.button("Compare documents", type="primary"):
        with st.spinner("Parsing documents, retrieving matches, and auditing mismatches..."):
            master_doc = parse_document(master_file, "master")
            service_doc = parse_document(service_file, "service")
            retrieval_results = retrieve_matches(master_doc, service_doc)
            predictions = run_auditor_and_guardrail(retrieval_results, master_doc["full_text"], service_doc["full_text"], pair_id=f"ui-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}")
        st.session_state["report"] = {
            "master_doc": master_doc,
            "service_doc": service_doc,
            "retrieval_results": retrieval_results,
            "predictions": predictions,
        }

    report = st.session_state.get("report")
    if report:
        render_results(report)


def render_results(report: dict) -> None:
    import streamlit as st

    predictions = sort_predictions_for_display(report["predictions"])
    positives = [p for p in predictions if p["has_contradiction"] or p["guardrail_action"] == "claim_withheld"]
    st.subheader("Audit results")
    st.write(f"Parsed {len(report['master_doc']['chunks'])} master chunk(s), {len(report['service_doc']['chunks'])} service chunk(s), and generated {len(predictions)} verified prediction(s).")

    json_bytes = json.dumps(report, indent=2).encode("utf-8")
    pdf_bytes = build_report_pdf(predictions)
    export_left, export_right = st.columns(2)
    export_left.download_button("Download full report JSON", json_bytes, file_name="crossdoc_report.json", mime="application/json")
    export_right.download_button("Download summary PDF", pdf_bytes, file_name="crossdoc_report.pdf", mime="application/pdf")

    if not positives:
        st.success("No contradictions were found by the offline stub auditor.")
        return

    for prediction in positives:
        render_prediction_card(prediction)


def render_prediction_card(prediction: dict) -> None:
    import streamlit as st

    category = infer_category(prediction)
    severity = prediction.get("severity", "None")
    with st.container(border=True):
        header_cols = st.columns([3, 1, 2])
        header_cols[0].markdown(f"### {category}")
        header_cols[1].markdown(_badge(severity, "severity"), unsafe_allow_html=True)
        header_cols[2].markdown(_verification_badge(prediction), unsafe_allow_html=True)

        if prediction.get("guardrail_action") == "claim_withheld":
            st.error("This claim could not be verified against the source text and has been withheld.")

        quote_cols = st.columns(2)
        quote_cols[0].markdown("**Master clause**")
        quote_cols[0].write(prediction.get("msa_exact_quote") or "—")
        quote_cols[1].markdown("**Service clause**")
        quote_cols[1].write(prediction.get("sow_exact_quote") or "—")
        st.markdown("**Explanation**")
        st.write(prediction.get("conflict_explanation") or "—")
        st.markdown("**Suggested redline**")
        st.write(prediction.get("suggested_redline") or "—")


def _badge(text: str, kind: str) -> str:
    colors = {"High": "#b42318", "Medium": "#b54708", "Low": "#175cd3", "None": "#667085"}
    color = colors.get(text, "#667085") if kind == "severity" else "#067647"
    return f"<span style='background:{color};color:white;padding:0.2rem 0.5rem;border-radius:999px;font-weight:700'>{text}</span>"


def _verification_badge(prediction: dict) -> str:
    if prediction.get("guardrail_action") == "claim_withheld":
        return "<span style='background:#b42318;color:white;padding:0.2rem 0.5rem;border-radius:999px;font-weight:700'>Claim withheld</span>"
    if prediction.get("guardrail_verified"):
        return "<span style='background:#067647;color:white;padding:0.2rem 0.5rem;border-radius:999px;font-weight:700'>Evidence verified</span>"
    return "<span style='background:#667085;color:white;padding:0.2rem 0.5rem;border-radius:999px;font-weight:700'>Unverified</span>"


if __name__ == "__main__":
    main()
