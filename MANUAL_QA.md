# Part 7 Manual QA Checklist

1. Install dependencies from `requirements.txt` in an environment with Streamlit available.
2. Run the app:

   ```bash
   streamlit run app.py
   ```

3. Upload `sample_docs/sample_master_msa.pdf` in the Master column.
4. Upload `sample_docs/sample_service_sow.pdf` in the Service column.
5. Click **Compare documents**.
6. Confirm the results list renders without an exception.
7. Confirm at least one visible mismatch card shows a severity badge and side-by-side Master/Service quotes.
8. Confirm at least one card displays: **"This claim could not be verified against the source text and has been withheld."**
9. Click **Download full report JSON** and confirm a JSON file downloads.
10. Click **Download summary PDF** and confirm a PDF file downloads.

Automated smoke coverage for the same stub pipeline is in `tests/test_part7_pipeline.py`.
