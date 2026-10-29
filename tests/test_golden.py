"""Golden invoice integration tests — real files processed through full pipeline with real LLM.

Run with:  pytest tests/test_golden.py -v
Requires: LLM API key (XAI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in env / .env)

Assertions target deterministic code checks (vendor lookup, fraud keywords,
arithmetic, duplicate detection) — not LLM decisions, which may vary between runs.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from setup_db import setup_database
from graph.workflow import process_invoice


INVOICES_DIR = os.path.join(os.path.dirname(__file__), "..", "invoices2")


def _inv(filename: str) -> str:
    return os.path.join(INVOICES_DIR, filename)


def _has_api_key() -> bool:
    return bool(
        os.environ.get("LLM_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


requires_llm = pytest.mark.skipif(
    not _has_api_key(),
    reason="No LLM API key — skipping integration test",
)


@pytest.fixture
def fresh_db():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "golden.db")
        setup_database(db_path)
        yield db_path


def _run(invoice_file: str, db_path: str):
    with patch("tools.inventory_db.DB_PATH", db_path):
        return process_invoice(invoice_file)


# ---------- invoice_2001.pdf — Scanned PDF (multimodal vision extraction) ----------

@requires_llm
class TestInvoice2001_ScannedPDF:

    def test_extraction_succeeds(self, fresh_db):
        """Scanned PDF should be rendered as images and extracted via LLM vision."""
        result = _run(_inv("invoice_2001.pdf"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"
        assert result.stage_reached == "completed"
        assert result.extraction is not None
        assert result.extraction.invoice.invoice_number is not None

    def test_vendor_check_runs(self, fresh_db):
        result = _run(_inv("invoice_2001.pdf"), fresh_db)
        assert result.validation is not None
        assert result.validation.vendor_check is not None


# ---------- credit_memo_2003.pdf — Credit note (-$1,000) ----------

@requires_llm
class TestCreditMemo2003:

    def test_vendor_recognized(self, fresh_db):
        result = _run(_inv("credit_memo_2003.pdf"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V007"
        assert vc.status == "active"

    def test_negative_total_extracted(self, fresh_db):
        result = _run(_inv("credit_memo_2003.pdf"), fresh_db)
        assert result.extraction.invoice.total < 0

    def test_credit_note_flagged(self, fresh_db):
        result = _run(_inv("credit_memo_2003.pdf"), fresh_db)
        credit_flags = [
            d for d in result.validation.data_issues
            if "credit" in d.lower()
        ]
        assert len(credit_flags) > 0, (
            f"Expected credit note flag in data_issues: {result.validation.data_issues}"
        )


# ---------- invoice_2004.txt — Fraud / social engineering ----------

@requires_llm
class TestInvoice2004_FraudLanguage:

    def test_vendor_is_known(self, fresh_db):
        result = _run(_inv("invoice_2004.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V004"
        assert vc.status == "active"

    def test_fraud_keywords_detected(self, fresh_db):
        result = _run(_inv("invoice_2004.txt"), fresh_db)
        assert len(result.validation.fraud_signals) > 0
        signals_text = " ".join(result.validation.fraud_signals).lower()
        assert "pre-approved" in signals_text or "pre approved" in signals_text

    def test_not_auto_approved(self, fresh_db):
        result = _run(_inv("invoice_2004.txt"), fresh_db)
        assert result.approval.decision in ("rejected", "needs_review")


# ---------- invoice_2005.pdf — EUR invoice (German RECHNUNG) ----------

@requires_llm
class TestInvoice2005_EUR:

    def test_vendor_matched_eur(self, fresh_db):
        result = _run(_inv("invoice_2005.pdf"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V010"
        assert vc.currency == "EUR"

    def test_usd_conversion(self, fresh_db):
        result = _run(_inv("invoice_2005.pdf"), fresh_db)
        assert result.validation.usd_total is not None
        assert result.validation.usd_total > 10_000

    def test_extraction_captures_eur(self, fresh_db):
        result = _run(_inv("invoice_2005.pdf"), fresh_db)
        assert result.extraction.invoice.currency == "EUR"


# ---------- invoice_2006.txt — Pressure language ----------

@requires_llm
class TestInvoice2006_PressureLanguage:

    def test_vendor_recognized(self, fresh_db):
        result = _run(_inv("invoice_2006.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V002"

    def test_pressure_language_flagged(self, fresh_db):
        result = _run(_inv("invoice_2006.txt"), fresh_db)
        assert len(result.validation.fraud_signals) > 0
        signals_text = " ".join(result.validation.fraud_signals).lower()
        assert "immediately" in signals_text


# ---------- invoice_2007.csv — Clean invoice (should approve) ----------

@requires_llm
class TestInvoice2007_Clean:

    def test_vendor_active(self, fresh_db):
        result = _run(_inv("invoice_2007.csv"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"
        assert result.stage_reached == "completed"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V005"
        assert vc.status == "active"

    def test_no_fraud_signals(self, fresh_db):
        result = _run(_inv("invoice_2007.csv"), fresh_db)
        assert len(result.validation.fraud_signals) == 0

    def test_no_duplicate(self, fresh_db):
        result = _run(_inv("invoice_2007.csv"), fresh_db)
        assert result.validation.is_duplicate is False

    def test_approved_and_paid(self, fresh_db):
        result = _run(_inv("invoice_2007.csv"), fresh_db)
        assert result.approval.decision in ("approved", "needs_review")
        if result.approval.decision == "approved":
            assert result.payment.status == "paid"


# ---------- invoice_2008.pdf — Multi-page, unknown items, >$10K ----------

@requires_llm
class TestInvoice2008_MultiPage:

    def test_vendor_recognized(self, fresh_db):
        result = _run(_inv("invoice_2008.pdf"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"
        assert result.stage_reached == "completed"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V006"

    def test_items_not_in_master(self, fresh_db):
        result = _run(_inv("invoice_2008.pdf"), fresh_db)
        unknown = [
            c for c in result.validation.inventory_checks
            if not c.found_in_inventory
        ]
        assert len(unknown) > 0

    def test_above_vp_threshold(self, fresh_db):
        result = _run(_inv("invoice_2008.pdf"), fresh_db)
        total = result.validation.usd_total or result.extraction.invoice.total
        assert total > 10_000


# ---------- invoice_3001 — Inactive vendor ----------

@requires_llm
class TestInvoice3001_InactiveVendor:

    def test_vendor_found_but_inactive(self, fresh_db):
        result = _run(_inv("invoice_3001_inactive_vendor.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V013"
        assert vc.status == "inactive"
        assert vc.issue is not None

    def test_not_auto_approved(self, fresh_db):
        result = _run(_inv("invoice_3001_inactive_vendor.txt"), fresh_db)
        assert result.approval.decision in ("rejected", "needs_review")


# ---------- invoice_3002 — Vendor on hold ----------

@requires_llm
class TestInvoice3002_HoldVendor:

    def test_vendor_found_but_on_hold(self, fresh_db):
        result = _run(_inv("invoice_3002_hold_vendor.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V014"
        assert vc.status == "hold"

    def test_not_auto_approved(self, fresh_db):
        result = _run(_inv("invoice_3002_hold_vendor.txt"), fresh_db)
        assert result.approval.decision in ("rejected", "needs_review")


# ---------- invoice_3003 — Arithmetic error ----------

@requires_llm
class TestInvoice3003_ArithmeticError:

    def test_arithmetic_error_detected(self, fresh_db):
        result = _run(_inv("invoice_3003_arithmetic_error.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"
        assert len(result.validation.arithmetic_errors) > 0

    def test_not_auto_approved(self, fresh_db):
        result = _run(_inv("invoice_3003_arithmetic_error.txt"), fresh_db)
        assert result.approval.decision in ("rejected", "needs_review")


# ---------- invoice_3004 — CFO threshold (>$50K) ----------

@requires_llm
class TestInvoice3004_CFOThreshold:

    def test_above_cfo_threshold(self, fresh_db):
        result = _run(_inv("invoice_3004_cfo_threshold.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        total = result.validation.usd_total or result.extraction.invoice.total
        assert total > 50_000

    def test_held_for_cfo(self, fresh_db):
        result = _run(_inv("invoice_3004_cfo_threshold.txt"), fresh_db)
        assert result.approval.decision == "needs_review"
        assert "cfo_required" in result.approval.flags
        assert result.payment.status == "held"


# ---------- invoice_3005 — Missing required fields ----------

@requires_llm
class TestInvoice3005_MissingFields:

    def test_data_issues_flagged(self, fresh_db):
        result = _run(_inv("invoice_3005_missing_fields.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"
        assert len(result.validation.data_issues) > 0

    def test_not_auto_approved(self, fresh_db):
        result = _run(_inv("invoice_3005_missing_fields.txt"), fresh_db)
        assert result.approval.decision in ("rejected", "needs_review")


# ---------- invoice_3006 — Pricing deviation >10% ----------

@requires_llm
class TestInvoice3006_PriceDeviation:

    def test_vendor_recognized(self, fresh_db):
        result = _run(_inv("invoice_3006_price_deviation.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"

        vc = result.validation.vendor_check
        assert vc.found_in_system is True
        assert vc.vendor_id == "V001"

    def test_price_deviation_flagged(self, fresh_db):
        """WidgetA at $350 is 40% above reference $250; GadgetX at $400 is 47% below $750."""
        result = _run(_inv("invoice_3006_price_deviation.txt"), fresh_db)
        assert result.validation.passed is False or len(result.validation.data_issues) > 0


# ---------- invoice_3007 — Zero quantity + negative unit price ----------

@requires_llm
class TestInvoice3007_ZeroQuantity:

    def test_data_issues_flagged(self, fresh_db):
        result = _run(_inv("invoice_3007_zero_quantity.txt"), fresh_db)
        assert result.error is None, f"Pipeline error: {result.error}"
        assert len(result.validation.data_issues) > 0

    def test_not_auto_approved(self, fresh_db):
        result = _run(_inv("invoice_3007_zero_quantity.txt"), fresh_db)
        assert result.approval.decision in ("rejected", "needs_review")


# ---------- Duplicate detection (process same invoice twice) ----------

@requires_llm
class TestDuplicateDetection:

    def test_second_run_rejected(self, fresh_db):
        r1 = _run(_inv("invoice_2007.csv"), fresh_db)
        assert r1.error is None
        assert r1.stage_reached == "completed"

        r2 = _run(_inv("invoice_2007.csv"), fresh_db)
        assert r2.error is None
        assert r2.validation.is_duplicate is True
        assert r2.approval.decision == "rejected"
        assert any("duplicate" in f.lower() for f in r2.approval.flags)
