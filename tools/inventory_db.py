"""ERP database tools — vendors, items, inventory, payments, FX rates."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH

_ABBREVS = {
    "incorporated": "inc", "inc.": "inc",
    "company": "co", "co.": "co",
    "corporation": "corp", "corp.": "corp",
    "limited": "ltd", "ltd.": "ltd",
    "&": "and",
}


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- Item / Inventory lookups (preserved from v1) ---

def check_item(item_name: str, db_path: str | None = None) -> dict:
    """Look up a single item in the inventory database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT item, stock FROM inventory WHERE item = ?",
        (item_name,),
    )
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return {"item": item_name, "found": False, "stock": None, "unit_price": None, "category": None}

    inv = dict(row)
    cursor.execute("SELECT unit_price, description FROM items WHERE item = ?", (item_name,))
    item_row = cursor.fetchone()
    conn.close()

    unit_price = dict(item_row)["unit_price"] if item_row else None
    return {
        "item": inv["item"],
        "found": True,
        "stock": inv["stock"],
        "unit_price": unit_price,
        "category": None,
    }


def check_items(item_names: list[str], db_path: str | None = None) -> list[dict]:
    """Look up multiple items in the inventory database."""
    return [check_item(name, db_path) for name in item_names]


def get_all_items(db_path: str | None = None) -> list[dict]:
    """Return all items in the inventory for fuzzy matching."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.item, inv.stock, i.unit_price
        FROM items i LEFT JOIN inventory inv ON i.item = inv.item
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Vendor lookups (new) ---

def normalize_vendor_name(name: str) -> str:
    """Normalize vendor name for matching: lowercase, strip punctuation, expand abbreviations."""
    n = name.lower().strip()
    for full, short in _ABBREVS.items():
        n = n.replace(full, short)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return " ".join(n.split())


def lookup_vendor(vendor_name: str, db_path: str | None = None) -> dict | None:
    """Find a vendor by name using normalized matching. Returns best match or None."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT vendor_id, legal_name, currency, payment_terms, status, onboarded_on "
        "FROM vendors"
    )
    rows = cursor.fetchall()
    conn.close()

    target = normalize_vendor_name(vendor_name)

    for row in rows:
        if normalize_vendor_name(dict(row)["legal_name"]) == target:
            return dict(row)
    for row in rows:
        db_norm = normalize_vendor_name(dict(row)["legal_name"])
        if target in db_norm or db_norm in target:
            return dict(row)
    return None


# --- Duplicate / payment tracking (new) ---

def normalize_invoice_number(inv_num: str) -> str:
    """Normalize invoice number for duplicate comparison: strip formatting."""
    n = re.sub(r"[^a-zA-Z0-9]", "", inv_num).upper()
    n = re.sub(r"^INV0*", "", n)
    return n


def check_duplicate_payment(
    invoice_number: str,
    vendor_name: str,
    db_path: str | None = None,
) -> dict:
    """Check if this invoice was already paid or processed."""
    norm_inv = normalize_invoice_number(invoice_number)
    norm_vendor = normalize_vendor_name(vendor_name)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT invoice_number, amount, paid_on FROM payments")
    for row in cursor.fetchall():
        r = dict(row)
        if normalize_invoice_number(r["invoice_number"]) == norm_inv:
            conn.close()
            return {"is_duplicate": True, "source": "payments", "details": r}

    cursor.execute(
        "SELECT invoice_number, vendor_name, decision, processed_on "
        "FROM processed_invoices"
    )
    for row in cursor.fetchall():
        r = dict(row)
        if normalize_invoice_number(r["invoice_number"]) == norm_inv and \
           normalize_vendor_name(r["vendor_name"]) == norm_vendor:
            conn.close()
            return {"is_duplicate": True, "source": "processed_invoices", "details": r}

    conn.close()
    return {"is_duplicate": False}


def record_processed_invoice(
    invoice_number: str,
    vendor_name: str,
    amount: float,
    decision: str,
    db_path: str | None = None,
) -> None:
    """Record a processed invoice to prevent duplicates within and across runs."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO processed_invoices VALUES (?, ?, ?, ?, ?)",
        (invoice_number, vendor_name, amount, decision,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def record_payment(
    invoice_number: str,
    vendor_name: str,
    amount: float,
    currency: str,
    vendor_id: str | None = None,
    db_path: str | None = None,
) -> str:
    """Record a payment in the payments table. Returns payment_id."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM payments")
    count = cursor.fetchone()[0]
    payment_id = f"PAY-{count + 1:04d}"
    cursor.execute(
        "INSERT INTO payments VALUES (?, ?, ?, ?, ?, ?)",
        (payment_id, vendor_id, invoice_number, amount, currency,
         datetime.now(timezone.utc).strftime("%Y-%m-%d")),
    )
    conn.commit()
    conn.close()
    return payment_id


# --- FX rates (new) ---

def get_fx_rate(
    currency: str,
    invoice_date: str | None = None,
    db_path: str | None = None,
) -> float | None:
    """Get the USD exchange rate for a currency on or before a date."""
    if currency == "USD":
        return 1.0
    conn = get_connection(db_path)
    cursor = conn.cursor()
    if invoice_date:
        cursor.execute(
            "SELECT usd_per_unit FROM fx_rates "
            "WHERE currency = ? AND rate_date <= ? "
            "ORDER BY rate_date DESC LIMIT 1",
            (currency, invoice_date),
        )
    else:
        cursor.execute(
            "SELECT usd_per_unit FROM fx_rates "
            "WHERE currency = ? ORDER BY rate_date DESC LIMIT 1",
            (currency,),
        )
    row = cursor.fetchone()
    conn.close()
    return dict(row)["usd_per_unit"] if row else None


def get_vendor_history(
    vendor_name: str,
    db_path: str | None = None,
) -> list[dict]:
    """Get processing history for a vendor from the processed_invoices table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    norm = normalize_vendor_name(vendor_name)
    cursor.execute("SELECT * FROM processed_invoices ORDER BY processed_on DESC")
    results = []
    for row in cursor.fetchall():
        r = dict(row)
        if normalize_vendor_name(r["vendor_name"]) == norm:
            results.append(r)
    conn.close()
    return results
