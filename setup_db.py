"""Initialize the SQLite ERP database with all tables and seed data.

Creates: vendors, items, inventory, payments, fx_rates, processed_invoices
All in a single database file.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "acme_erp.db")


def setup_database(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF")

    for table in [
        "processed_invoices", "fx_rates", "payments",
        "inventory", "items", "vendors",
    ]:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    cursor.execute("""
        CREATE TABLE vendors (
            vendor_id      TEXT PRIMARY KEY,
            legal_name     TEXT NOT NULL,
            currency       TEXT NOT NULL DEFAULT 'USD',
            payment_terms  TEXT,
            status         TEXT NOT NULL CHECK (status IN ('active', 'hold', 'inactive')),
            onboarded_on   TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE items (
            item         TEXT PRIMARY KEY,
            description  TEXT NOT NULL,
            unit_price   REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE inventory (
            item   TEXT PRIMARY KEY,
            stock  INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE payments (
            payment_id      TEXT PRIMARY KEY,
            vendor_id       TEXT REFERENCES vendors(vendor_id),
            invoice_number  TEXT NOT NULL,
            amount          REAL NOT NULL,
            currency        TEXT NOT NULL DEFAULT 'USD',
            paid_on         TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE fx_rates (
            currency      TEXT NOT NULL,
            rate_date     TEXT NOT NULL,
            usd_per_unit  REAL NOT NULL,
            PRIMARY KEY (currency, rate_date)
        )
    """)

    cursor.execute("""
        CREATE TABLE processed_invoices (
            invoice_number  TEXT NOT NULL,
            vendor_name     TEXT NOT NULL,
            amount          REAL,
            decision        TEXT,
            processed_on    TEXT NOT NULL,
            PRIMARY KEY (invoice_number, vendor_name)
        )
    """)

    # --- Seed data ---

    cursor.executemany(
        "INSERT INTO vendors VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("V001", "Widgets Inc.", "USD", "Net 15", "active", "2019-03-04"),
            ("V002", "Gadgets Co.", "USD", "Net 30", "active", "2020-06-15"),
            ("V003", "Precision Parts Ltd.", "USD", "Net 30", "active", "2021-01-11"),
            ("V004", "Acme Industrial Supplies", "USD", "Net 15", "active", "2018-09-30"),
            ("V005", "MegaWidgets Corp", "USD", "Net 30", "active", "2022-02-01"),
            ("V006", "Consolidated Materials Group", "USD", "Net 30", "active", "2017-05-22"),
            ("V007", "Summit Manufacturing Co.", "USD", "Net 30", "active", "2020-10-05"),
            ("V008", "QuickShip Distributers", "USD", "Net 30", "active", "2019-11-18"),
            ("V009", "Atlas Industrial Supply", "USD", "Net 60", "active", "2016-04-12"),
            ("V010", "TechParts International", "EUR", "Net 30", "active", "2023-03-01"),
            ("V011", "Reliable Components Inc.", "USD", "Net 30", "active", "2025-12-01"),
            ("V012", "Global Supply Chain Partners", "USD", "Net 60", "active", "2022-08-15"),
            ("V013", "Sunset Supplies LLC", "USD", "Net 30", "inactive", "2018-01-10"),
            ("V014", "Pending Partners Inc.", "USD", "Net 30", "hold", "2024-06-01"),
        ],
    )

    cursor.executemany(
        "INSERT INTO items VALUES (?, ?, ?)",
        [
            ("WidgetA", "Widget, type A", 250.00),
            ("WidgetB", "Widget, type B", 500.00),
            ("GadgetX", "Gadget, model X", 750.00),
            ("FakeItem", "Legacy test item — do not order", 1000.00),
        ],
    )

    cursor.executemany(
        "INSERT INTO inventory VALUES (?, ?)",
        [
            ("WidgetA", 15),
            ("WidgetB", 10),
            ("GadgetX", 5),
            ("FakeItem", 0),
        ],
    )

    cursor.executemany(
        "INSERT INTO fx_rates VALUES (?, ?, ?)",
        [
            ("EUR", "2025-12-01", 1.16),
            ("EUR", "2026-01-01", 1.17),
            ("EUR", "2026-02-01", 1.18),
        ],
    )

    conn.commit()
    conn.close()
    print(f"ERP database initialized at {db_path}")


if __name__ == "__main__":
    setup_database()
