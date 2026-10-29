"""Validation agent — verifies extracted invoice data against ERP database and flags issues."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from models.schemas import (
    ExtractionResult,
    InventoryCheck,
    ValidationResult,
    VendorCheck,
)
from tools.inventory_db import (
    check_duplicate_payment,
    check_item,
    get_all_items,
    get_fx_rate,
    lookup_vendor,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a validation agent for invoice processing. You receive pre-computed check \
results from code-based lookups (vendor status, inventory, duplicates, arithmetic). \
Your job is to analyze ALL these results together and produce a final risk assessment.

Evaluate:
1. VENDOR: Is the vendor known and active in the vendor master? Unknown or inactive \
   vendors are serious red flags.
2. ITEMS: Are items recognized in the item master? Unknown items need investigation.
3. DATA INTEGRITY: Missing required fields (vendor, invoice number, date, items, total), \
   negative quantities, arithmetic errors.
4. PRICING: Compare invoiced unit prices against reference prices from the item master. \
   Flag significant deviations.
5. FRAUD SIGNALS: Pressure language, suspicious vendor names, instructions to skip review, \
   unrealistic amounts, items with zero stock.
6. DUPLICATES: Has this invoice already been paid or processed?

IMPORTANT:
- Stock mismatches (requested qty > available stock) are a legitimate concern — they \
  may indicate items not yet received, over-billing, or a delivery discrepancy. Flag \
  them as data issues. Note that warehouse snapshots can lag receipts, but significant \
  mismatches should increase risk score.
- Arithmetic must add up: each line = qty × unit_price, subtotal = sum of lines, \
  total = subtotal + tax. Allow $0.01 rounding tolerance.
- If the vendor is NOT found in the vendor master, this is a high-risk flag.
- If the invoice is a duplicate (already paid), mark is_duplicate=true and set high risk.

RISK SCORING (0.0 = safe, 1.0 = critical):
- 0.0-0.2: Clean invoice, minor issues at most
- 0.2-0.4: Some flags but likely legitimate
- 0.4-0.6: Material issues requiring attention
- 0.6-0.8: Serious concerns, likely should reject
- 0.8-1.0: Critical — probable fraud or major data problems

Respond with valid JSON matching the ValidationResult schema."""

FRAUD_KEYWORDS = [
    "urgent", "immediately", "wire transfer", "pay now",
    "pre-approved", "skip review", "asap", "do not route",
    "process without po", "pre approved", "already approved",
    "bypass", "no review needed", "rush payment",
]


def run_validation(extraction: ExtractionResult) -> dict[str, Any]:
    """Validate extracted invoice data against ERP database and business rules.

    Flow: code runs deterministic checks → results passed to LLM → LLM reasons over evidence.
    """
    invoice = extraction.invoice

    # --- Code-based checks (deterministic, no LLM) ---
    vendor_check = _validate_vendor(invoice.vendor)
    inventory_checks = _validate_items(invoice)
    data_issues = _check_data_integrity(invoice, extraction)
    arithmetic_errors = _check_arithmetic(invoice)
    pricing_flags = _check_pricing(invoice)
    fraud_signals = _check_fraud_signals(invoice, extraction, inventory_checks)
    duplicate = check_duplicate_payment(invoice.invoice_number, invoice.vendor or "")
    usd_total = _compute_usd_total(invoice)

    all_items = get_all_items()
    known_items = {item["item"] for item in all_items}

    # --- LLM risk assessment (receives all check results as context) ---
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ValidationResult)

    context = _build_context(
        invoice, extraction, vendor_check, inventory_checks,
        data_issues, arithmetic_errors, pricing_flags, fraud_signals,
        duplicate, known_items, usd_total,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    result: ValidationResult = structured_llm.invoke(messages)

    # Override LLM output with code-computed values for consistency
    result.vendor_check = vendor_check
    result.inventory_checks = inventory_checks
    result.arithmetic_errors = arithmetic_errors
    result.is_duplicate = duplicate["is_duplicate"]
    result.usd_total = usd_total

    # Add pricing deviations to data_issues deterministically
    for flag in pricing_flags:
        if flag not in result.data_issues:
            result.data_issues.append(flag)

    # Add stock mismatches to data_issues deterministically
    for check in inventory_checks:
        if check.issue and check.issue not in result.data_issues:
            result.data_issues.append(check.issue)

    needs_correction = _needs_re_extraction(result, extraction, known_items)
    correction_feedback = ""
    if needs_correction:
        correction_feedback = _build_correction_feedback(result, known_items)

    logger.info(
        "Validation complete: %s (passed=%s, risk=%.2f, flags=%d)",
        invoice.invoice_number,
        result.passed,
        result.risk_score,
        len(result.fraud_signals) + len(result.data_issues),
    )

    return {
        "validation": result,
        "needs_correction": needs_correction,
        "correction_feedback": correction_feedback,
    }


# --- Vendor validation ---

def _validate_vendor(vendor_name: str | None) -> VendorCheck:
    if not vendor_name or vendor_name.strip() == "":
        return VendorCheck(
            vendor_name=vendor_name or "",
            found_in_system=False,
            issue="Missing vendor name",
        )
    match = lookup_vendor(vendor_name)
    if match is None:
        return VendorCheck(
            vendor_name=vendor_name,
            found_in_system=False,
            issue=f"Vendor '{vendor_name}' not found in vendor master",
        )
    if match["status"] == "inactive":
        return VendorCheck(
            vendor_name=vendor_name,
            found_in_system=True,
            vendor_id=match["vendor_id"],
            status=match["status"],
            currency=match["currency"],
            issue=f"Vendor '{vendor_name}' is INACTIVE — reject",
        )
    if match["status"] == "hold":
        return VendorCheck(
            vendor_name=vendor_name,
            found_in_system=True,
            vendor_id=match["vendor_id"],
            status=match["status"],
            currency=match["currency"],
            issue=f"Vendor '{vendor_name}' is ON HOLD — route to Vendor Management",
        )
    return VendorCheck(
        vendor_name=vendor_name,
        found_in_system=True,
        vendor_id=match["vendor_id"],
        status=match["status"],
        currency=match["currency"],
    )


# --- Item / inventory validation ---

def _validate_items(invoice) -> list[InventoryCheck]:
    checks = []
    for line in invoice.items:
        db_result = check_item(line.item_name)
        check = InventoryCheck(
            item_name=line.item_name,
            found_in_inventory=db_result["found"],
            available_stock=db_result["stock"],
            requested_quantity=line.quantity,
            stock_sufficient=(
                db_result["stock"] >= line.quantity
                if db_result["found"] and db_result["stock"] is not None
                else None
            ),
        )
        if not db_result["found"]:
            check.issue = f"Item '{line.item_name}' not found in item master"
        elif db_result["stock"] == 0:
            check.issue = f"Item '{line.item_name}' has zero stock"
        elif db_result["stock"] is not None and db_result["stock"] < line.quantity:
            check.issue = (
                f"Requested {line.quantity} of '{line.item_name}' "
                f"but only {db_result['stock']} in stock"
            )
        checks.append(check)
    return checks


# --- Data integrity ---

def _check_data_integrity(invoice, extraction: ExtractionResult) -> list[str]:
    issues = []
    if not invoice.vendor or invoice.vendor.strip() == "":
        issues.append("REQUIRED: Missing vendor name")
    if not invoice.invoice_number:
        issues.append("REQUIRED: Missing invoice number")
    if not invoice.date:
        issues.append("REQUIRED: Missing invoice date")
    if not invoice.items:
        issues.append("REQUIRED: No line items")
    for line in invoice.items:
        if line.quantity <= 0:
            issues.append(
                f"INVALID: Quantity {line.quantity} for {line.item_name} "
                f"(must be positive)"
            )
        if line.unit_price < 0:
            issues.append(f"INVALID: Negative unit price for {line.item_name}")
    if invoice.total < 0:
        issues.append(f"CREDIT NOTE: Negative total amount {invoice.total} — verify this is a legitimate credit memo")
    return issues


# --- Arithmetic validation (programmatic, $0.01 tolerance) ---

def _check_arithmetic(invoice) -> list[str]:
    errors = []
    tolerance = 0.01

    for line in invoice.items:
        expected = line.quantity * line.unit_price
        if line.line_total is not None and abs(line.line_total - expected) > tolerance:
            errors.append(
                f"Line '{line.item_name}': stated {line.line_total:.2f} "
                f"!= {line.quantity} x {line.unit_price:.2f} = {expected:.2f}"
            )

    computed_subtotal = sum(
        (line.line_total if line.line_total is not None else line.quantity * line.unit_price)
        for line in invoice.items
    )

    if invoice.subtotal is not None and abs(computed_subtotal - invoice.subtotal) > tolerance:
        errors.append(
            f"Subtotal: stated {invoice.subtotal:.2f} != sum of lines {computed_subtotal:.2f}"
        )

    if invoice.subtotal is not None and invoice.tax is not None:
        expected_total = invoice.subtotal + invoice.tax
        if abs(invoice.total - expected_total) > tolerance:
            errors.append(
                f"Total: stated {invoice.total:.2f} "
                f"!= subtotal({invoice.subtotal:.2f}) + tax({invoice.tax:.2f}) "
                f"= {expected_total:.2f}"
            )

    return errors


# --- Pricing validation (compare against item master reference prices) ---

def _check_pricing(invoice) -> list[str]:
    flags = []
    for line in invoice.items:
        db_result = check_item(line.item_name)
        if db_result["found"] and db_result["unit_price"] is not None:
            ref_price = db_result["unit_price"]
            if ref_price > 0:
                deviation = abs(line.unit_price - ref_price) / ref_price
                if deviation > 0.10:
                    flags.append(
                        f"Price deviation for '{line.item_name}': "
                        f"invoiced {line.unit_price:.2f} vs reference {ref_price:.2f} "
                        f"({deviation:.0%} off)"
                    )
    return flags


# --- Fraud detection ---

def _check_fraud_signals(
    invoice, extraction: ExtractionResult, checks: list[InventoryCheck]
) -> list[str]:
    signals = []
    text_to_scan = " ".join([
        invoice.vendor or "",
        invoice.notes or "",
        invoice.payment_terms or "",
        " ".join(extraction.raw_anomalies),
        " ".join(extraction.issues),
    ]).lower()
    for kw in FRAUD_KEYWORDS:
        if kw in text_to_scan:
            signals.append(f"Possible fraud — language detected: '{kw}'")

    vendor_lower = (invoice.vendor or "").lower()
    if any(word in vendor_lower for word in ["fraud", "fake", "scam", "test"]):
        signals.append(f"Suspicious vendor name: '{invoice.vendor}'")

    zero_stock_items = [
        c for c in checks if c.available_stock == 0 and c.found_in_inventory
    ]
    if zero_stock_items:
        signals.append(
            f"Items with zero stock ordered: {[c.item_name for c in zero_stock_items]}"
        )

    if invoice.total > 100_000:
        signals.append(f"Unusually high invoice amount: {invoice.total}")

    return signals


# --- FX conversion ---

def _compute_usd_total(invoice) -> float | None:
    if invoice.currency == "USD":
        return invoice.total
    rate = get_fx_rate(invoice.currency, invoice.date)
    if rate is not None:
        return round(invoice.total * rate, 2)
    return None


# --- Context builder for LLM ---

def _build_context(
    invoice, extraction, vendor_check, inventory_checks,
    data_issues, arithmetic_errors, pricing_flags, fraud_signals,
    duplicate, known_items, usd_total,
) -> str:
    lines = [
        f"Invoice: {invoice.invoice_number}",
        f"Vendor: {invoice.vendor}",
        f"Date: {invoice.date}",
        f"Total: {invoice.currency} {invoice.total}",
        f"USD equivalent: {usd_total}",
        f"Extraction confidence: {extraction.confidence}",
        f"Extraction issues: {extraction.issues}",
        "",
        "=== VENDOR CHECK (from vendor master DB) ===",
        f"  Found in system: {vendor_check.found_in_system}",
        f"  Vendor ID: {vendor_check.vendor_id}",
        f"  Status: {vendor_check.status}",
        f"  Registered currency: {vendor_check.currency}",
        f"  Issue: {vendor_check.issue}",
        "",
        "=== ITEM CHECKS (from item master + warehouse DB) ===",
    ]
    for c in inventory_checks:
        lines.append(
            f"  - {c.item_name}: in_item_master={c.found_in_inventory}, "
            f"warehouse_stock={c.available_stock}, requested={c.requested_quantity}, "
            f"issue={c.issue}"
        )
    lines.extend([
        "",
        f"=== DATA INTEGRITY ISSUES === {data_issues}",
        f"=== ARITHMETIC ERRORS === {arithmetic_errors}",
        f"=== PRICING FLAGS === {pricing_flags}",
        f"=== FRAUD SIGNALS === {fraud_signals}",
        f"=== DUPLICATE CHECK === is_duplicate={duplicate['is_duplicate']}, details={duplicate}",
        f"=== KNOWN ITEMS === {sorted(known_items)}",
    ])
    return "\n".join(lines)


# --- Self-correction logic ---

def _needs_re_extraction(
    result: ValidationResult,
    extraction: ExtractionResult,
    known_items: set[str],
) -> bool:
    for check in result.inventory_checks:
        if not check.found_in_inventory:
            normalized = check.item_name.replace(" ", "").replace("-", "").replace("_", "")
            for known in known_items:
                if normalized.lower() == known.lower() and check.item_name != known:
                    return True
    return False


def _build_correction_feedback(result: ValidationResult, known_items: set[str]) -> str:
    feedback_parts = ["The following items may have been incorrectly extracted:"]
    for check in result.inventory_checks:
        if not check.found_in_inventory:
            feedback_parts.append(
                f"- '{check.item_name}' not found. Known items: {sorted(known_items)}. "
                f"Check if this is a misspelling or naming variation."
            )
    return "\n".join(feedback_parts)
