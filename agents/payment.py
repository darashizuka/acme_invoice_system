"""Payment agent — processes approved invoices with duplicate prevention and DB recording."""

from __future__ import annotations

import logging
from typing import Any

from models.schemas import (
    ApprovalDecision,
    ExtractionResult,
    PaymentResult,
    ValidationResult,
)
from tools.inventory_db import (
    check_duplicate_payment,
    record_payment,
    record_processed_invoice,
)
from tools.payment_api import mock_payment

logger = logging.getLogger(__name__)


def run_payment(
    extraction: ExtractionResult,
    approval: ApprovalDecision,
    validation: ValidationResult | None = None,
) -> dict[str, Any]:
    """Process payment for approved invoices or log rejection/hold.

    Includes duplicate prevention check before payment and records
    all outcomes to the database for audit trail.
    """
    invoice = extraction.invoice
    vendor_id = None
    if validation and validation.vendor_check:
        vendor_id = validation.vendor_check.vendor_id

    if approval.decision == "approved":
        dup = check_duplicate_payment(invoice.invoice_number, invoice.vendor or "")
        if dup["is_duplicate"]:
            logger.warning(
                "Duplicate payment blocked for %s: %s",
                invoice.invoice_number, dup,
            )
            payment = PaymentResult(
                status="rejected",
                vendor=invoice.vendor,
                amount=invoice.total,
                currency=invoice.currency,
                reason=(
                    f"BLOCKED: Duplicate payment detected. "
                    f"Source: {dup.get('source')}, details: {dup.get('details')}"
                ),
            )
            return {"payment": payment}

        result = mock_payment(
            vendor=invoice.vendor,
            amount=invoice.total,
            currency=invoice.currency,
        )

        payment_id = record_payment(
            invoice_number=invoice.invoice_number,
            vendor_name=invoice.vendor,
            amount=invoice.total,
            currency=invoice.currency,
            vendor_id=vendor_id,
        )
        record_processed_invoice(
            invoice.invoice_number, invoice.vendor,
            invoice.total, "paid",
        )

        payment = PaymentResult(
            status="paid",
            vendor=invoice.vendor,
            amount=invoice.total,
            currency=invoice.currency,
            reason=f"Approved and paid (ref: {result['reference']}, db: {payment_id})",
        )
        logger.info(
            "Payment sent: %s %s to %s (ref: %s, db: %s)",
            invoice.currency, invoice.total, invoice.vendor,
            result["reference"], payment_id,
        )

    elif approval.decision == "rejected":
        payment = PaymentResult(
            status="rejected",
            vendor=invoice.vendor,
            amount=invoice.total,
            currency=invoice.currency,
            reason=f"Rejected: {approval.reasoning}",
        )
        record_processed_invoice(
            invoice.invoice_number, invoice.vendor,
            invoice.total, "rejected",
        )
        logger.info(
            "Payment rejected for %s: %s",
            invoice.invoice_number, approval.reasoning[:100],
        )

    else:
        payment = PaymentResult(
            status="held",
            vendor=invoice.vendor,
            amount=invoice.total,
            currency=invoice.currency,
            reason=f"Held for review: {approval.reasoning}",
        )
        record_processed_invoice(
            invoice.invoice_number, invoice.vendor,
            invoice.total, "held",
        )
        logger.info(
            "Payment held for %s: %s",
            invoice.invoice_number, approval.reasoning[:100],
        )

    return {"payment": payment}
