"""CLI entry point for the invoice processing pipeline.

Usage:
    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --batch=data/invoices
    python main.py --invoice_path=data/invoices/invoice_1003.txt --verbose
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from graph.workflow import process_invoice
from models.schemas import PipelineResult
from setup_db import setup_database

console = Console(force_terminal=True, soft_wrap=True)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def render_result(result: PipelineResult, verbose: bool = False) -> None:
    """Render a single invoice result with Rich formatting."""
    if result.error:
        console.print(Panel(
            f"[red bold]ERROR:[/] {result.error}\n"
            f"Stage reached: {result.stage_reached}",
            title=f"[red]{result.file_path}[/]",
            border_style="red",
        ))
        return

    inv = result.extraction.invoice if result.extraction else None
    inv_num = inv.invoice_number if inv else "Unknown"
    vendor = inv.vendor if inv else "Unknown"

    decision = result.approval.decision if result.approval else "N/A"
    decision_color = {"approved": "green", "rejected": "red", "needs_review": "yellow"}.get(
        decision, "white"
    )

    payment_status = result.payment.status if result.payment else "N/A"
    payment_color = {"paid": "green", "rejected": "red", "held": "yellow"}.get(
        payment_status, "white"
    )

    header = Text()
    header.append(f"{inv_num}", style="bold")
    header.append(f" — {vendor}")
    header.append(f"  |  ", style="dim")
    header.append(f"{decision.upper()}", style=f"bold {decision_color}")
    header.append(f"  →  ", style="dim")
    header.append(f"{payment_status.upper()}", style=f"bold {payment_color}")

    lines = []

    if result.extraction:
        ext = result.extraction
        lines.append(f"[bold]Extraction[/] (confidence: {ext.confidence:.0%})")
        lines.append(f"  Total: {inv.currency} {inv.total:,.2f}  |  Due: {inv.due_date or 'N/A'}")
        if inv.items:
            item_table = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
            item_table.add_column("Item", style="cyan")
            item_table.add_column("Qty", justify="right")
            item_table.add_column("Price", justify="right")
            for item in inv.items:
                item_table.add_row(
                    item.item_name,
                    str(item.quantity),
                    f"{inv.currency} {item.unit_price:,.2f}",
                )
            console.print(item_table)
        if ext.issues:
            lines.append(f"  Issues: {', '.join(ext.issues)}")

    if result.validation:
        val = result.validation
        risk_color = "green" if val.risk_score < 0.3 else "yellow" if val.risk_score < 0.6 else "red"
        lines.append(f"\n[bold]Validation[/] (risk: [{risk_color}]{val.risk_score:.0%}[/])")
        if val.data_issues:
            for issue in val.data_issues:
                lines.append(f"  [yellow]⚠ {issue}[/]")
        if val.fraud_signals:
            for signal in val.fraud_signals:
                lines.append(f"  [red]🚩 {signal}[/]")
        for check in val.inventory_checks:
            if check.issue:
                lines.append(f"  [yellow]• {check.issue}[/]")

    if result.approval:
        appr = result.approval
        lines.append(f"\n[bold]Approval[/] ({appr.risk_level})")
        lines.append(f"  Reasoning: {appr.reasoning}")
        if appr.reflection:
            lines.append(f"  Reflection: {appr.reflection}")

    if result.payment:
        pay = result.payment
        lines.append(f"\n[bold]Payment[/]")
        lines.append(f"  {pay.reason or 'Processed'}")

    if result.audit_log and verbose:
        lines.append(f"\n[bold]Audit Trail[/]")
        for i, entry in enumerate(result.audit_log):
            elapsed = ""
            if i > 0 and entry.timestamp and result.audit_log[i - 1].timestamp:
                dt = entry.timestamp - result.audit_log[i - 1].timestamp
                elapsed = f" [dim]({dt:.1f}s)[/]"
            lines.append(f"  [dim]{entry.stage}:[/] {entry.detail}{elapsed}")
        total_time = result.audit_log[-1].timestamp - result.audit_log[0].timestamp
        lines.append(f"  [dim]Total: {total_time:.1f}s[/]")

    console.print(Panel(
        "\n".join(lines),
        title=str(header),
        border_style=decision_color,
        padding=(1, 2),
    ))


def render_summary(results: list[PipelineResult]) -> None:
    """Render a summary table for batch processing."""
    table = Table(title="Invoice Processing Summary", show_lines=True)
    table.add_column("Invoice", style="cyan")
    table.add_column("Vendor")
    table.add_column("Total", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("Decision", justify="center")
    table.add_column("Payment", justify="center")

    totals = {"approved": 0, "rejected": 0, "needs_review": 0, "error": 0}

    for r in results:
        if r.error:
            table.add_row(
                os.path.basename(r.file_path),
                "—", "—", "—",
                "[red]ERROR[/]",
                "—",
            )
            totals["error"] += 1
            continue

        inv = r.extraction.invoice if r.extraction else None
        decision = r.approval.decision if r.approval else "N/A"
        risk = r.validation.risk_score if r.validation else 0
        payment = r.payment.status if r.payment else "N/A"

        risk_color = "green" if risk < 0.3 else "yellow" if risk < 0.6 else "red"
        dec_color = {"approved": "green", "rejected": "red", "needs_review": "yellow"}.get(decision, "white")
        pay_color = {"paid": "green", "rejected": "red", "held": "yellow"}.get(payment, "white")

        table.add_row(
            inv.invoice_number if inv else "?",
            (inv.vendor if inv else "?")[:25],
            f"{inv.currency} {inv.total:,.2f}" if inv else "?",
            f"[{risk_color}]{risk:.0%}[/]",
            f"[{dec_color}]{decision}[/]",
            f"[{pay_color}]{payment}[/]",
        )
        if decision in totals:
            totals[decision] += 1

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/] "
        f"[green]{totals['approved']} approved[/]  "
        f"[red]{totals['rejected']} rejected[/]  "
        f"[yellow]{totals['needs_review']} held[/]  "
        f"[red]{totals['error']} errors[/]"
    )


def export_results(results: list[PipelineResult], output_path: str) -> None:
    """Export results as JSON Lines."""
    with open(output_path, "w") as f:
        for r in results:
            record = {
                "source_file": os.path.basename(r.file_path),
                "stage_reached": r.stage_reached,
                "error": r.error,
            }
            if r.extraction:
                record["invoice"] = r.extraction.invoice.model_dump()
                record["confidence"] = r.extraction.confidence
                record["extraction_issues"] = r.extraction.issues
            if r.validation:
                record["risk_score"] = r.validation.risk_score
                record["validation_passed"] = r.validation.passed
                record["data_issues"] = r.validation.data_issues
                record["fraud_signals"] = r.validation.fraud_signals
            if r.approval:
                record["decision"] = r.approval.decision
                record["risk_level"] = r.approval.risk_level
                record["reasoning"] = r.approval.reasoning
                record["reflection"] = r.approval.reflection
            if r.payment:
                record["payment_status"] = r.payment.status
                record["payment_reason"] = r.payment.reason
            f.write(json.dumps(record) + "\n")
    console.print(f"\n[dim]Results exported to {output_path}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Galatiq Invoice Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py --invoice_path=data/invoices/invoice_1001.txt\n"
               "  python main.py --batch=data/invoices\n"
               "  python main.py --batch=data/invoices --out=results.jsonl\n",
    )
    parser.add_argument("--invoice_path", type=str, help="Path to a single invoice file")
    parser.add_argument("--batch", type=str, help="Directory of invoices to process in batch")
    parser.add_argument("--out", type=str, help="Output path for JSON Lines results")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--db", type=str, help="Path to inventory database (default: inventory.db)")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.invoice_path and not args.batch:
        parser.print_help()
        sys.exit(1)

    db_path = args.db or os.path.join(os.path.dirname(__file__), "acme_erp.db")
    if not os.path.exists(db_path):
        console.print("[yellow]Inventory database not found — initializing...[/]")
        setup_database(db_path)

    console.print(Panel(
        "[bold]Galatiq Invoice Processing Pipeline[/]\n"
        "Multi-agent system: Ingestion → Validation → Approval → Payment",
        border_style="blue",
    ))

    results: list[PipelineResult] = []

    if args.invoice_path:
        if not os.path.exists(args.invoice_path):
            console.print(f"[red]File not found: {args.invoice_path}[/]")
            sys.exit(1)
        console.print(f"\nProcessing: [cyan]{args.invoice_path}[/]\n")
        result = process_invoice(args.invoice_path)
        results.append(result)
        render_result(result, verbose=args.verbose)

    elif args.batch:
        if not os.path.isdir(args.batch):
            console.print(f"[red]Directory not found: {args.batch}[/]")
            sys.exit(1)

        files = sorted(glob.glob(os.path.join(args.batch, "*")))
        invoice_files = [f for f in files if os.path.splitext(f)[1].lower() in {".txt", ".json", ".csv", ".pdf", ".xml"}]

        seen_bases = set()
        deduped = []
        for f in invoice_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base in seen_bases:
                continue
            seen_bases.add(base)
            deduped.append(f)
        invoice_files = deduped

        console.print(f"\nBatch processing [cyan]{len(invoice_files)}[/] invoices from {args.batch}\n")

        for i, fpath in enumerate(invoice_files, 1):
            console.print(f"[dim]({i}/{len(invoice_files)})[/] {os.path.basename(fpath)}")
            result = process_invoice(fpath)
            results.append(result)
            render_result(result, verbose=args.verbose)
            console.print()

        render_summary(results)

    if args.out and results:
        export_results(results, args.out)


if __name__ == "__main__":
    main()
