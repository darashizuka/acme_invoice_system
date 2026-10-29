"""Mock pipeline tests — full workflow routing with mocked LLM responses.

These tests verify pipeline logic (auto-approve, CFO hold, safety overrides,
duplicate blocking, self-correction) without any LLM API calls.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import (
    ApprovalDecision,
    ExtractionResult,
    InvoiceData,
    LineItem,
    ValidationResult,
    VendorCheck,
)
from setup_db import setup_database


# --- Mock LLM helper ---

class MockStructuredLLM:
    """Mock that returns predefined Pydantic objects based on schema type."""

    def __init__(self, responses: dict):
        self._responses = responses
        self._call_counts = {}

    def with_structured_output(self, schema):
        mock = MagicMock()
        schema_name = schema.__name__
        self._call_counts.setdefault(schema_name, 0)

        def invoke_side_effect(messages):
            response_list = self._responses.get(schema_name, [])
            idx = self._call_counts[schema_name]
            self._call_counts[schema_name] = idx + 1
            if isinstance(response_list, list):
                return response_list[idx % len(response_list)]
            return response_list

        mock.invoke = MagicMock(side_effect=invoke_side_effect)
        return mock


def _make_mock_llm(
    extraction=None,
    validation=None,
    approval=None,
):
    """Build a MockStructuredLLM with predefined responses per schema."""
    if extraction is None:
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-1001",
                vendor="Widgets Inc.",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=10, unit_price=250.0, line_total=2500.0)],
                total=2500.0,
                currency="USD",
            ),
            confidence=0.95,
        )
    if validation is None:
        validation = ValidationResult(
            passed=True,
            risk_score=0.1,
            summary="Clean invoice",
        )
    if approval is None:
        approval = ApprovalDecision(
            decision="approved",
            reasoning="Looks good",
            reflection="Confirmed",
            risk_level="low",
        )

    responses = {
        "ExtractionResult": [extraction],
        "ValidationResult": [validation],
        "ApprovalDecision": [approval, approval],
    }
    return MockStructuredLLM(responses)


# --- Fixtures ---

@pytest.fixture
def test_env():
    """Create a temp dir with a test DB and a simple invoice file."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test.db")
        setup_database(db_path)

        inv_path = os.path.join(d, "test_invoice.txt")
        with open(inv_path, "w") as f:
            f.write("Invoice: INV-1001\nVendor: Widgets Inc.\nWidgetA x10 @ $250\nTotal: $2,500.00")

        yield {"db_path": db_path, "inv_path": inv_path, "tmp_dir": d}


def _run_pipeline(test_env, mock_llm):
    """Run process_invoice with mocked LLM and test DB."""
    from graph.workflow import process_invoice
    with patch("agents.ingestion.get_llm", return_value=mock_llm), \
         patch("agents.validation.get_llm", return_value=mock_llm), \
         patch("agents.approval.get_llm", return_value=mock_llm), \
         patch("tools.inventory_db.DB_PATH", test_env["db_path"]):
        return process_invoice(test_env["inv_path"])


# --- Test cases ---

class TestAutoApprove:
    """Clean invoice under $10K → auto-approved, paid."""

    def test_clean_small_invoice(self, test_env):
        mock = _make_mock_llm(
            validation=ValidationResult(
                passed=True, risk_score=0.1, summary="Clean",
                vendor_check=VendorCheck(
                    vendor_name="Widgets Inc.", found_in_system=True,
                    vendor_id="V001", status="active", currency="USD",
                ),
            ),
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.stage_reached == "completed"
        assert result.approval.decision == "approved"
        assert result.approval.flags == []
        assert result.payment.status == "paid"


class TestVPReview:
    """Invoice $10K-$50K → VP review path."""

    def test_medium_invoice_vp_review(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-2001",
                vendor="Gadgets Co.",
                date="2026-01-15",
                items=[LineItem(item_name="GadgetX", quantity=30, unit_price=750.0, line_total=22500.0)],
                total=22500.0,
                currency="USD",
            ),
            confidence=0.92,
        )
        approval = ApprovalDecision(
            decision="approved",
            reasoning="VP review: legitimate order from known vendor",
            reflection="Confirmed after reflection",
            risk_level="medium",
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=True, risk_score=0.2, summary="Clean but above VP threshold",
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.approval.decision == "approved"
        assert result.payment.status == "paid"


class TestCFOHold:
    """Invoice >$50K → held for CFO co-approval."""

    def test_large_invoice_cfo_hold(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-3001",
                vendor="Widgets Inc.",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=300, unit_price=250.0, line_total=75000.0)],
                total=75000.0,
                currency="USD",
            ),
            confidence=0.95,
        )
        approval = ApprovalDecision(
            decision="approved",
            reasoning="VP would approve",
            reflection="Looks good to VP",
            risk_level="medium",
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=True, risk_score=0.15, summary="Clean but large",
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.approval.decision == "needs_review"
        assert "cfo_required" in result.approval.flags
        assert result.payment.status == "held"


class TestDuplicateBlocked:
    """Second processing of same invoice → payment blocked."""

    def test_duplicate_payment_blocked(self, test_env):
        mock = _make_mock_llm(
            validation=ValidationResult(
                passed=True, risk_score=0.1, summary="Clean",
                vendor_check=VendorCheck(
                    vendor_name="Widgets Inc.", found_in_system=True,
                    vendor_id="V001", status="active", currency="USD",
                ),
            ),
        )
        # First run — should succeed
        result1 = _run_pipeline(test_env, mock)
        assert result1.payment.status == "paid"

        # Second run — same invoice, should be blocked
        mock2 = _make_mock_llm(
            validation=ValidationResult(
                passed=False, risk_score=0.9, summary="Duplicate",
                is_duplicate=True,
            ),
        )
        result2 = _run_pipeline(test_env, mock2)
        assert result2.approval.decision == "rejected"
        assert result2.payment.status == "rejected"


class TestUnknownVendor:
    """Unknown vendor + LLM tries to approve → safety override catches it."""

    def test_unknown_vendor_safety_override(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-4001",
                vendor="Fraudster LLC",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=5, unit_price=250.0, line_total=1250.0)],
                total=1250.0,
                currency="USD",
            ),
            confidence=0.90,
        )
        approval = ApprovalDecision(
            decision="approved",
            reasoning="LLM incorrectly approves",
            reflection="Looks fine",
            risk_level="low",
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=False, risk_score=0.7, summary="Unknown vendor",
                vendor_check=VendorCheck(
                    vendor_name="Fraudster LLC", found_in_system=False,
                    issue="Not in vendor master",
                ),
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.approval.decision == "needs_review"
        assert "safety_override" in result.approval.flags


class TestFraudRejection:
    """Fraud signals → LLM rejects."""

    def test_fraud_signals_rejected(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-5001",
                vendor="Widgets Inc.",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=5, unit_price=250.0, line_total=1250.0)],
                total=1250.0,
                currency="USD",
                notes="URGENT: Pay immediately! Pre-approved, skip review.",
            ),
            confidence=0.85,
        )
        approval = ApprovalDecision(
            decision="rejected",
            reasoning="Multiple fraud signals detected",
            reflection="Confirmed rejection — pressure language",
            risk_level="critical",
            flags=["fraud_language"],
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=False, risk_score=0.85, summary="Fraud signals",
                fraud_signals=["urgent", "pre-approved", "skip review"],
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.approval.decision == "rejected"
        assert result.payment.status == "rejected"


class TestEURConversion:
    """EUR invoice → USD conversion for threshold comparison."""

    def test_eur_invoice_conversion(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-6001",
                vendor="TechParts International",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=5, unit_price=200.0, line_total=1000.0)],
                total=1000.0,
                currency="EUR",
            ),
            confidence=0.90,
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=True, risk_score=0.15, summary="Clean EUR invoice",
                usd_total=1170.0,
                vendor_check=VendorCheck(
                    vendor_name="TechParts International", found_in_system=True,
                    vendor_id="V010", status="active", currency="EUR",
                ),
            ),
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.stage_reached == "completed"


class TestParseError:
    """Unsupported file format → parse error, pipeline stops."""

    def test_unsupported_format(self, test_env):
        bad_path = os.path.join(test_env["tmp_dir"], "invoice.docx")
        with open(bad_path, "w") as f:
            f.write("test")

        test_env_copy = dict(test_env)
        test_env_copy["inv_path"] = bad_path

        mock = _make_mock_llm()
        result = _run_pipeline(test_env_copy, mock)

        assert result.error is not None
        assert "Unsupported" in result.error
        assert result.stage_reached == "error"
        assert result.extraction is None


class TestCreditNote:
    """Negative total (credit note) flows through without INVALID flag."""

    def test_credit_note_not_invalid(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-7001",
                vendor="Widgets Inc.",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=5, unit_price=250.0, line_total=1250.0)],
                total=-1250.0,
                currency="USD",
            ),
            confidence=0.90,
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=True, risk_score=0.3, summary="Credit note",
            ),
            approval=ApprovalDecision(
                decision="needs_review",
                reasoning="Credit note requires manual verification",
                reflection="Hold for review",
                risk_level="medium",
            ),
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None


class TestInactiveVendor:
    """Inactive vendor → not auto-approved, routed to VP review."""

    def test_inactive_vendor_not_auto_approved(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-8001",
                vendor="Sunset Supplies LLC",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=5, unit_price=250.0, line_total=1250.0)],
                total=1250.0,
                currency="USD",
            ),
            confidence=0.90,
        )
        approval = ApprovalDecision(
            decision="rejected",
            reasoning="Vendor is inactive",
            reflection="Confirmed rejection",
            risk_level="high",
            flags=["inactive_vendor"],
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=False, risk_score=0.6, summary="Inactive vendor",
                vendor_check=VendorCheck(
                    vendor_name="Sunset Supplies LLC", found_in_system=True,
                    vendor_id="V013", status="inactive", currency="USD",
                    issue="Vendor is INACTIVE — reject",
                ),
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.approval.decision in ("rejected", "needs_review")


class TestHoldVendor:
    """Vendor on hold → not auto-approved."""

    def test_hold_vendor_not_auto_approved(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-8002",
                vendor="Pending Partners Inc.",
                date="2026-01-15",
                items=[LineItem(item_name="GadgetX", quantity=3, unit_price=750.0, line_total=2250.0)],
                total=2250.0,
                currency="USD",
            ),
            confidence=0.90,
        )
        approval = ApprovalDecision(
            decision="needs_review",
            reasoning="Vendor is on hold",
            reflection="Hold for vendor management",
            risk_level="medium",
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=False, risk_score=0.5, summary="Vendor on hold",
                vendor_check=VendorCheck(
                    vendor_name="Pending Partners Inc.", found_in_system=True,
                    vendor_id="V014", status="hold", currency="USD",
                    issue="Vendor is ON HOLD — route to Vendor Management",
                ),
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.approval.decision == "needs_review"
        assert result.payment.status == "held"


class TestArithmeticError:
    """Arithmetic errors → not auto-approved, held for review."""

    def test_arithmetic_error_held(self, test_env):
        extraction = ExtractionResult(
            invoice=InvoiceData(
                invoice_number="INV-8003",
                vendor="Widgets Inc.",
                date="2026-01-15",
                items=[LineItem(item_name="WidgetA", quantity=10, unit_price=250.0, line_total=2500.0)],
                total=2800.0,
                currency="USD",
            ),
            confidence=0.85,
        )
        approval = ApprovalDecision(
            decision="needs_review",
            reasoning="Arithmetic error — total doesn't match line items",
            reflection="Hold for manual verification",
            risk_level="medium",
        )
        mock = _make_mock_llm(
            extraction=extraction,
            validation=ValidationResult(
                passed=False, risk_score=0.4, summary="Arithmetic error",
                arithmetic_errors=["Total $2,800.00 != sum of line items $2,500.00"],
                vendor_check=VendorCheck(
                    vendor_name="Widgets Inc.", found_in_system=True,
                    vendor_id="V001", status="active", currency="USD",
                ),
            ),
            approval=approval,
        )
        result = _run_pipeline(test_env, mock)

        assert result.error is None
        assert result.approval.decision == "needs_review"
        assert result.payment.status == "held"
