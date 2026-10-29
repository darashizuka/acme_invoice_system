"""Tests for ERP database tools — vendor matching, duplicate detection, FX rates."""

import os
import tempfile

import pytest

from setup_db import setup_database
from tools.inventory_db import (
    check_duplicate_payment,
    check_item,
    check_items,
    get_all_items,
    get_fx_rate,
    lookup_vendor,
    normalize_invoice_number,
    normalize_vendor_name,
    record_payment,
    record_processed_invoice,
)


@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test.db")
        setup_database(db_path)
        yield db_path


# --- Vendor name normalization ---

class TestNormalizeVendorName:
    def test_basic(self):
        assert normalize_vendor_name("Widgets Inc.") == "widgets inc"

    def test_case_insensitive(self):
        assert normalize_vendor_name("WIDGETS INC") == "widgets inc"

    def test_abbreviation_expansion(self):
        assert normalize_vendor_name("Acme Incorporated") == "acme inc"
        assert normalize_vendor_name("Acme Corp.") == "acme corp"
        assert normalize_vendor_name("Acme Limited") == "acme ltd"
        assert normalize_vendor_name("Acme Company") == "acme co"

    def test_strip_punctuation(self):
        assert normalize_vendor_name("Acme, Inc.") == "acme inc"

    def test_ampersand(self):
        assert normalize_vendor_name("Smith & Jones") == "smith and jones"


# --- Vendor lookup ---

class TestLookupVendor:
    def test_exact_match(self, test_db):
        result = lookup_vendor("Widgets Inc.", test_db)
        assert result is not None
        assert result["vendor_id"] == "V001"
        assert result["status"] == "active"

    def test_case_insensitive_match(self, test_db):
        result = lookup_vendor("widgets inc.", test_db)
        assert result is not None
        assert result["vendor_id"] == "V001"

    def test_partial_match(self, test_db):
        result = lookup_vendor("Gadgets", test_db)
        assert result is not None
        assert result["vendor_id"] == "V002"

    def test_unknown_vendor(self, test_db):
        result = lookup_vendor("Fraudster LLC", test_db)
        assert result is None

    def test_eur_vendor(self, test_db):
        result = lookup_vendor("TechParts International", test_db)
        assert result is not None
        assert result["currency"] == "EUR"


# --- Item checks ---

class TestCheckItem:
    def test_found(self, test_db):
        result = check_item("WidgetA", test_db)
        assert result["found"] is True
        assert result["stock"] == 15
        assert result["unit_price"] == 250.00

    def test_not_found(self, test_db):
        result = check_item("UnknownItem", test_db)
        assert result["found"] is False
        assert result["stock"] is None

    def test_check_items_batch(self, test_db):
        results = check_items(["WidgetA", "GadgetX", "FakeItem"], test_db)
        assert len(results) == 3
        assert results[0]["found"] is True
        assert results[2]["stock"] == 0

    def test_get_all_items(self, test_db):
        items = get_all_items(test_db)
        assert len(items) == 4
        names = {i["item"] for i in items}
        assert "WidgetA" in names
        assert "GadgetX" in names


# --- Invoice number normalization ---

class TestNormalizeInvoiceNumber:
    def test_standard(self):
        assert normalize_invoice_number("INV-0974") == "974"

    def test_with_spaces(self):
        assert normalize_invoice_number("INV 974") == "974"

    def test_hash_prefix(self):
        assert normalize_invoice_number("#974") == "974"

    def test_bare_number(self):
        assert normalize_invoice_number("974") == "974"

    def test_all_same(self):
        results = {
            normalize_invoice_number(v)
            for v in ["INV-0974", "INV 974", "#974", "974"]
        }
        assert len(results) == 1


# --- Duplicate detection ---

class TestDuplicatePayment:
    def test_no_duplicate(self, test_db):
        result = check_duplicate_payment("INV-9999", "Acme Corp", test_db)
        assert result["is_duplicate"] is False

    def test_duplicate_after_recording(self, test_db):
        record_payment("INV-1001", "Widgets Inc.", 500.0, "USD", "V001", test_db)
        result = check_duplicate_payment("INV-1001", "Widgets Inc.", test_db)
        assert result["is_duplicate"] is True
        assert result["source"] == "payments"

    def test_duplicate_normalized(self, test_db):
        record_payment("INV-0042", "Acme Co.", 1000.0, "USD", db_path=test_db)
        result = check_duplicate_payment("INV 42", "Acme Co.", test_db)
        assert result["is_duplicate"] is True

    def test_processed_invoice_duplicate(self, test_db):
        record_processed_invoice("INV-2000", "Gadgets Co.", 750.0, "paid", test_db)
        result = check_duplicate_payment("INV-2000", "Gadgets Co.", test_db)
        assert result["is_duplicate"] is True
        assert result["source"] == "processed_invoices"


# --- FX rates ---

class TestFxRate:
    def test_usd(self, test_db):
        rate = get_fx_rate("USD", "2026-01-15", test_db)
        assert rate == 1.0

    def test_eur_with_date(self, test_db):
        rate = get_fx_rate("EUR", "2026-01-15", test_db)
        assert rate == 1.17

    def test_eur_latest(self, test_db):
        rate = get_fx_rate("EUR", None, test_db)
        assert rate == 1.18

    def test_unknown_currency(self, test_db):
        rate = get_fx_rate("GBP", "2026-01-15", test_db)
        assert rate is None
