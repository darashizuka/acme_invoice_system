"""Tests for deterministic approval logic (tier routing, safety overrides)."""

import pytest

from models.schemas import (
    ApprovalDecision,
    ExtractionResult,
    InvoiceData,
    LineItem,
    ValidationResult,
    VendorCheck,
)
from agents.approval import _should_override, _apply_override


def _make_validation(**overrides):
    defaults = {
        "passed": True,
        "risk_score": 0.1,
        "summary": "Clean invoice",
        "is_duplicate": False,
        "vendor_check": VendorCheck(
            vendor_name="Widgets Inc.",
            found_in_system=True,
            vendor_id="V001",
            status="active",
            currency="USD",
        ),
    }
    defaults.update(overrides)
    return ValidationResult(**defaults)


def _make_decision(**overrides):
    defaults = {
        "decision": "approved",
        "reasoning": "Looks good",
        "reflection": "Confirmed",
        "risk_level": "low",
        "flags": [],
    }
    defaults.update(overrides)
    return ApprovalDecision(**defaults)


# --- Safety overrides ---

class TestSafetyOverride:
    """Safety overrides only trigger on deterministic facts: duplicate, unknown vendor."""

    def test_no_override_clean(self):
        val = _make_validation()
        dec = _make_decision()
        assert _should_override(dec, val) is False

    def test_no_override_when_rejected(self):
        val = _make_validation(is_duplicate=True)
        dec = _make_decision(decision="rejected")
        assert _should_override(dec, val) is False

    def test_no_override_high_risk(self):
        """High risk score is LLM judgment — not a deterministic override."""
        val = _make_validation(risk_score=0.9)
        dec = _make_decision(decision="approved")
        assert _should_override(dec, val) is False

    def test_no_override_fraud_signals(self):
        """Fraud signals are LLM judgment — not a deterministic override."""
        val = _make_validation(fraud_signals=["Pressure language detected"])
        dec = _make_decision(decision="approved")
        assert _should_override(dec, val) is False

    def test_override_duplicate(self):
        val = _make_validation(is_duplicate=True)
        dec = _make_decision(decision="approved")
        assert _should_override(dec, val) is True

    def test_override_unknown_vendor(self):
        val = _make_validation(
            vendor_check=VendorCheck(
                vendor_name="Unknown LLC",
                found_in_system=False,
            )
        )
        dec = _make_decision(decision="approved")
        assert _should_override(dec, val) is True

    def test_apply_override_unknown_vendor(self):
        val = _make_validation(
            vendor_check=VendorCheck(
                vendor_name="Unknown LLC",
                found_in_system=False,
            )
        )
        dec = _make_decision(decision="approved")
        result = _apply_override(dec, val)
        assert result.decision == "needs_review"
        assert result.risk_level == "critical"
        assert "safety_override" in result.flags
        assert "SAFETY OVERRIDE" in result.reflection

    def test_duplicate_override_rejects(self):
        val = _make_validation(is_duplicate=True)
        dec = _make_decision(decision="approved")
        result = _apply_override(dec, val)
        assert result.decision == "rejected"
        assert "duplicate_rejected" in result.flags
