"""Mock payment API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_payment_ledger: list[dict] = []


def mock_payment(vendor: str, amount: float, currency: str = "USD") -> dict:
    """Process a mock payment and record it in the ledger."""
    record = {
        "status": "success",
        "vendor": vendor,
        "amount": amount,
        "currency": currency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference": f"PAY-{len(_payment_ledger) + 1:04d}",
    }
    _payment_ledger.append(record)
    logger.info("Payment processed: %s to %s (%s %s)", record["reference"], vendor, currency, amount)
    return record


def get_ledger() -> list[dict]:
    """Return all payment records."""
    return list(_payment_ledger)


def clear_ledger() -> None:
    """Clear the payment ledger (for testing)."""
    _payment_ledger.clear()
