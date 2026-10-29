"""Tests for deterministic validation logic (no LLM calls)."""

import pytest

from models.schemas import (
    ExtractionResult,
    InvoiceData,
    LineItem,
)
from agents.validation import (
    _check_arithmetic,
    _check_data_integrity,
    _check_fraud_signals,
    _check_pricing,
    _validate_vendor,
    _validate_items,
)


def _make_invoice(**overrides):
    defaults = {
        "invoice_number": "INV-1001",
        "vendor": "Widgets Inc.",
        "date": "2026-01-15",
        "items": [LineItem(item_name="WidgetA", quantity=10, unit_price=250.0, line_total=2500.0)],
        "total": 2500.0,
        "currency": "USD",
    }
    defaults.update(overrides)
    return InvoiceData(**defaults)


def _make_extraction(invoice=None, **overrides):
    if invoice is None:
        invoice = _make_invoice()
    defaults = {
        "invoice": invoice,
        "confidence": 0.95,
        "issues": [],
        "raw_anomalies": [],
    }
    defaults.update(overrides)
    return ExtractionResult(**defaults)


# --- Arithmetic validation ---

class TestArithmetic:
    def test_clean_invoice(self):
        inv = _make_invoice(
            items=[LineItem(item_name="WidgetA", quantity=10, unit_price=250.0, line_total=2500.0)],
            subtotal=2500.0, tax=200.0, total=2700.0,
        )
        errors = _check_arithmetic(inv)
        assert errors == []

    def test_line_total_mismatch(self):
        inv = _make_invoice(
            items=[LineItem(item_name="WidgetA", quantity=10, unit_price=250.0, line_total=9999.0)],
        )
        errors = _check_arithmetic(inv)
        assert len(errors) == 1
        assert "WidgetA" in errors[0]

    def test_subtotal_mismatch(self):
        inv = _make_invoice(
            items=[LineItem(item_name="WidgetA", quantity=10, unit_price=250.0, line_total=2500.0)],
            subtotal=5000.0,
        )
        errors = _check_arithmetic(inv)
        assert len(errors) == 1
        assert "Subtotal" in errors[0]

    def test_total_mismatch(self):
        inv = _make_invoice(
            subtotal=2500.0, tax=200.0, total=9999.0,
        )
        errors = _check_arithmetic(inv)
        assert any("Total" in e for e in errors)

    def test_rounding_tolerance(self):
        inv = _make_invoice(
            items=[LineItem(item_name="WidgetA", quantity=3, unit_price=33.33, line_total=99.99)],
            subtotal=99.99, tax=8.00, total=107.99,
        )
        errors = _check_arithmetic(inv)
        assert errors == []


# --- Data integrity ---

class TestDataIntegrity:
    def test_clean(self):
        inv = _make_invoice()
        ext = _make_extraction(inv)
        issues = _check_data_integrity(inv, ext)
        assert issues == []

    def test_missing_vendor(self):
        inv = _make_invoice(vendor="")
        ext = _make_extraction(inv)
        issues = _check_data_integrity(inv, ext)
        assert any("Missing vendor" in i for i in issues)

    def test_negative_quantity(self):
        inv = _make_invoice(
            items=[LineItem(item_name="WidgetA", quantity=-5, unit_price=250.0)],
        )
        ext = _make_extraction(inv)
        issues = _check_data_integrity(inv, ext)
        assert any("Quantity" in i for i in issues)

    def test_negative_total_credit_note(self):
        inv = _make_invoice(total=-500.0)
        ext = _make_extraction(inv)
        issues = _check_data_integrity(inv, ext)
        assert any("CREDIT NOTE" in i for i in issues)
        assert not any("INVALID" in i for i in issues)


# --- Fraud signals ---

class TestFraudSignals:
    def test_clean(self):
        inv = _make_invoice()
        ext = _make_extraction(inv)
        signals = _check_fraud_signals(inv, ext, [])
        assert signals == []

    def test_urgent_language(self):
        inv = _make_invoice(notes="URGENT: Pay immediately!")
        ext = _make_extraction(inv)
        signals = _check_fraud_signals(inv, ext, [])
        assert any("urgent" in s for s in signals)
        assert any("immediately" in s for s in signals)

    def test_suspicious_vendor_name(self):
        inv = _make_invoice(vendor="Fraudster LLC")
        ext = _make_extraction(inv)
        signals = _check_fraud_signals(inv, ext, [])
        assert any("Suspicious vendor" in s for s in signals)

    def test_high_amount(self):
        inv = _make_invoice(total=150_000)
        ext = _make_extraction(inv)
        signals = _check_fraud_signals(inv, ext, [])
        assert any("Unusually high" in s for s in signals)

    def test_skip_review_language(self):
        inv = _make_invoice(notes="This has been pre-approved, skip review")
        ext = _make_extraction(inv)
        signals = _check_fraud_signals(inv, ext, [])
        assert any("pre-approved" in s or "pre approved" in s for s in signals)
        assert any("skip review" in s for s in signals)


# --- Vendor validation (uses test DB setup from conftest if available, else skips) ---

class TestVendorValidation:
    def test_missing_vendor(self):
        result = _validate_vendor(None)
        assert result.found_in_system is False
        assert "Missing" in result.issue

    def test_empty_vendor(self):
        result = _validate_vendor("")
        assert result.found_in_system is False
