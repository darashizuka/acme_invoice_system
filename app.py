"""Streamlit dashboard for the invoice processing pipeline.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import logging
import sqlite3
import tempfile

import streamlit as st

from graph.workflow import process_invoice
from models.schemas import PipelineResult
from setup_db import setup_database

logging.basicConfig(level=logging.INFO)

DB_PATH = os.path.join(os.path.dirname(__file__), "acme_erp.db")

st.set_page_config(
    page_title="AutoInvoice",
    layout="wide",
)

# -- Colors --
C_PRIMARY = "#1e3a5f"
C_ACCENT = "#7c6b58"
C_BG_WARM = "#faf8f5"
C_BG_CARD = "#ffffff"
C_BORDER = "#e8e4df"
C_TEXT = "#2d2a26"
C_TEXT_MUTED = "#8a8580"
C_GREEN = "#2d8a4e"
C_RED = "#c0392b"
C_AMBER = "#d4890a"
C_PURPLE = "#6e56cf"

STATUS_COLORS = {
    "paid": C_GREEN, "approved": C_GREEN,
    "rejected": C_RED, "duplicate_blocked": C_RED,
    "held": C_AMBER, "needs_review": C_AMBER,
}

STATUS_LABELS = {
    "paid": "Paid", "approved": "Approved",
    "rejected": "Rejected", "duplicate_blocked": "Rejected",
    "held": "On hold", "needs_review": "On hold",
}

# -- Global CSS --
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.main .block-container {{
    padding-top: 2rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {C_BG_WARM};
    border-right: 1px solid {C_BORDER};
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}

/* Brand */
.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 32px;
    padding: 0 4px;
}}
.brand-icon {{
    width: 32px;
    height: 32px;
    background: {C_PRIMARY};
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 16px;
    font-family: 'Inter', sans-serif;
}}
.brand-name {{
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: {C_PRIMARY};
    letter-spacing: -0.3px;
}}
.brand-name span {{
    font-weight: 400;
    color: {C_ACCENT};
}}

/* Section headers */
.section-title {{
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: {C_TEXT};
    margin-bottom: 4px;
}}
.section-sub {{
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: {C_TEXT_MUTED};
    margin-bottom: 16px;
}}
.section-label {{
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: {C_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    margin-top: 16px;
}}

/* Pipeline card */
.pipeline-card {{
    background: {C_BG_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}}
.pipeline-stages {{
    display: flex;
    align-items: center;
    gap: 0;
    justify-content: space-between;
}}
.pipeline-stage {{
    text-align: center;
    flex: 1;
}}
.pipeline-stage .count {{
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: {C_TEXT};
}}
.pipeline-stage .name {{
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: {C_TEXT_MUTED};
    margin-top: 2px;
}}
.pipeline-arrow {{
    color: {C_BORDER};
    font-size: 18px;
    padding: 0 4px;
}}

/* Metric card */
.metric-card {{
    background: {C_BG_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}}
.metric-card .value {{
    font-size: 26px;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
}}
.metric-card .label {{
    font-size: 11px;
    color: {C_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
    font-family: 'Inter', sans-serif;
}}

/* Invoice table */
.inv-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
}}
.inv-table th {{
    font-size: 11px;
    font-weight: 600;
    color: {C_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid {C_BORDER};
}}
.inv-table td {{
    padding: 14px 12px;
    border-bottom: 1px solid #f3f1ee;
    font-size: 14px;
    color: {C_TEXT};
    vertical-align: middle;
}}
.inv-table tr:hover {{
    background: {C_BG_WARM};
}}

/* Avatar */
.avatar {{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: white;
    margin-right: 10px;
    flex-shrink: 0;
}}
.vendor-cell {{
    display: flex;
    align-items: center;
}}
.vendor-name {{
    font-weight: 500;
}}
.vendor-date {{
    font-size: 12px;
    color: {C_TEXT_MUTED};
}}

/* Status badge */
.status-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}}

/* Stage pills */
.stage-pill {{
    text-align: center;
    padding: 10px 4px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
}}
.stage-done {{ background: #e8f5e9; color: {C_GREEN}; }}
.stage-active {{ background: #e8eaf6; color: {C_PURPLE}; border: 2px solid {C_PURPLE}; }}
.stage-pending {{ background: #f5f3f0; color: {C_TEXT_MUTED}; }}
.stage-error {{ background: #fce8e6; color: {C_RED}; }}

/* Decision banner */
.decision-banner {{
    padding: 14px 20px;
    border-left: 4px solid;
    border-radius: 4px;
    margin-bottom: 16px;
    font-family: 'Inter', sans-serif;
}}

/* Nav items */
.nav-item {{
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    padding: 8px 12px;
    border-radius: 8px;
    color: {C_TEXT};
    margin-bottom: 2px;
    cursor: default;
}}
.nav-active {{
    background: white;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
</style>
""", unsafe_allow_html=True)


def _escape_latex(text: str) -> str:
    return text.replace("$", "\\$")


def _get_initials(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return name[:2].upper() if name else "?"


AVATAR_COLORS = [
    "#1e3a5f", "#7c6b58", "#6e56cf", "#c0392b",
    "#2d8a4e", "#d4890a", "#4a7c59", "#8b5e3c",
]


def _avatar_color(name: str) -> str:
    return AVATAR_COLORS[hash(name) % len(AVATAR_COLORS)]


def ensure_db():
    if not os.path.exists(DB_PATH):
        setup_database(DB_PATH)


def get_recent_invoices(limit: int = 20) -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT invoice_number, vendor_name, amount, decision, processed_on "
        "FROM processed_invoices ORDER BY processed_on DESC LIMIT ?",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_pipeline_counts() -> dict:
    if not os.path.exists(DB_PATH):
        return {"total": 0, "paid": 0, "rejected": 0, "held": 0}
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM processed_invoices")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM processed_invoices WHERE decision = 'paid'")
    paid = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM processed_invoices WHERE decision = 'rejected'")
    rejected = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM processed_invoices WHERE decision IN ('held', 'needs_review')")
    held = cursor.fetchone()[0]
    conn.close()
    return {"total": total, "paid": paid, "rejected": rejected, "held": held}


# -- Render functions --

def render_pipeline_stages(result: PipelineResult):
    stages = ["Parse", "Ingest", "Validate", "Approve", "Pay"]
    stage_map = {
        "init": 0, "parsed": 1, "ingested": 2,
        "validated": 3, "approved": 4, "completed": 5, "error": -1,
    }
    reached = stage_map.get(result.stage_reached, 0)

    cols = st.columns(len(stages))
    for i, (col, stage_name) in enumerate(zip(cols, stages)):
        with col:
            if result.error and reached == -1:
                cls = "stage-error" if i == 0 else "stage-pending"
            elif i < reached:
                cls = "stage-done"
            elif i == reached:
                cls = "stage-active"
            else:
                cls = "stage-pending"
            st.markdown(
                f"<div class='stage-pill {cls}'>{stage_name}</div>",
                unsafe_allow_html=True,
            )


def render_extraction(result: PipelineResult):
    if not result.extraction:
        return
    ext = result.extraction
    inv = ext.invoice

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Invoice #", inv.invoice_number)
    col2.metric("Vendor", inv.vendor[:20] if inv.vendor else "Unknown")
    col3.metric("Total", f"{inv.currency} {inv.total:,.2f}")
    col4.metric("Confidence", f"{ext.confidence:.0%}")

    if inv.items:
        st.markdown("<div class='section-label'>Line Items</div>", unsafe_allow_html=True)
        items_data = []
        for item in inv.items:
            items_data.append({
                "Item": item.item_name,
                "Qty": item.quantity,
                "Unit Price": f"{inv.currency} {item.unit_price:,.2f}",
                "Line Total": f"{inv.currency} {(item.line_total or item.quantity * item.unit_price):,.2f}",
            })
        st.table(items_data)

    if ext.issues or ext.raw_anomalies:
        with st.expander("Issues & Anomalies"):
            for issue in ext.issues:
                st.warning(_escape_latex(issue))
            for anomaly in ext.raw_anomalies:
                st.info(_escape_latex(f"Anomaly: {anomaly}"))


def render_validation(result: PipelineResult):
    if not result.validation:
        return
    val = result.validation

    risk_color = C_GREEN if val.risk_score < 0.3 else C_AMBER if val.risk_score < 0.6 else C_RED

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='value' style='color:{risk_color};'>{val.risk_score:.0%}</div>"
            f"<div class='label'>Risk Score</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        c = C_GREEN if val.passed else C_RED
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='value' style='color:{c};'>{'Passed' if val.passed else 'Failed'}</div>"
            f"<div class='label'>Validation</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        c = C_RED if val.is_duplicate else C_GREEN
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='value' style='color:{c};'>{'Yes' if val.is_duplicate else 'No'}</div>"
            f"<div class='label'>Duplicate</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if val.vendor_check:
        vc = val.vendor_check
        with st.expander("Vendor Check", expanded=bool(vc.issue)):
            st.markdown(
                f"**{vc.vendor_name}** -- "
                f"{'Found' if vc.found_in_system else 'NOT FOUND'} in vendor master"
            )
            if vc.found_in_system:
                st.markdown(
                    f"  ID: `{vc.vendor_id}` | Status: **{vc.status}** | Currency: {vc.currency}"
                )
            if vc.issue:
                st.warning(_escape_latex(vc.issue))

    if val.inventory_checks:
        with st.expander("Inventory Checks", expanded=bool(any(c.issue for c in val.inventory_checks))):
            for check in val.inventory_checks:
                msg = f"**{check.item_name}**: "
                if check.found_in_inventory:
                    msg += f"Stock: {check.available_stock}, Requested: {check.requested_quantity}"
                else:
                    msg += "NOT FOUND in item master"
                if check.issue:
                    msg += f" -- {check.issue}"
                st.markdown(_escape_latex(msg))

    if val.arithmetic_errors:
        with st.expander("Arithmetic Errors", expanded=True):
            for err in val.arithmetic_errors:
                st.error(_escape_latex(err))

    if val.fraud_signals:
        with st.expander("Fraud Signals", expanded=True):
            for signal in val.fraud_signals:
                st.error(_escape_latex(signal))

    if val.data_issues:
        with st.expander("Data Issues"):
            for issue in val.data_issues:
                st.warning(_escape_latex(issue))


def render_approval(result: PipelineResult):
    if not result.approval:
        return
    appr = result.approval
    color = STATUS_COLORS.get(appr.decision, C_TEXT_MUTED)

    label = appr.decision.upper()
    if appr.decision == "needs_review":
        label = "HELD FOR REVIEW"

    st.markdown(
        f"<div class='decision-banner' style='background-color:{color}15;"
        f"border-color:{color};'>"
        f"<span style='font-size:18px;font-weight:700;color:{color};'>{label}</span>"
        f"<span style='margin-left:16px;color:{C_TEXT_MUTED};font-size:14px;'>"
        f"Risk level: {appr.risk_level}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Reasoning & Reflection", expanded=True):
        st.markdown(f"**Reasoning:** {_escape_latex(appr.reasoning)}")
        if appr.reflection:
            st.markdown(f"**Reflection:** {_escape_latex(appr.reflection)}")
        if appr.flags:
            st.markdown(f"**Flags:** {', '.join(appr.flags)}")


def render_payment(result: PipelineResult):
    if not result.payment:
        return
    pay = result.payment
    color = STATUS_COLORS.get(pay.status, C_TEXT_MUTED)

    st.markdown(
        f"<div class='decision-banner' style='background-color:{color}15;"
        f"border-color:{color};'>"
        f"<strong>Payment:</strong> {pay.status.upper()} -- "
        f"{pay.currency} {pay.amount:,.2f} to {pay.vendor}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if pay.reason:
        st.markdown(
            f"<div style='font-size:13px;color:{C_TEXT_MUTED};line-height:1.5;'>"
            f"{_escape_latex(pay.reason)}</div>",
            unsafe_allow_html=True,
        )


def render_result(result: PipelineResult):
    st.markdown("<div class='section-label'>Pipeline</div>", unsafe_allow_html=True)
    render_pipeline_stages(result)
    st.divider()

    if result.error:
        st.error(f"Pipeline error: {_escape_latex(result.error)}")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Extraction", "Validation", "Approval", "Payment",
    ])
    with tab1:
        render_extraction(result)
    with tab2:
        render_validation(result)
    with tab3:
        render_approval(result)
    with tab4:
        render_payment(result)

    if result.audit_log:
        with st.expander("Audit Trail"):
            for i, entry in enumerate(result.audit_log):
                elapsed = ""
                if i > 0 and entry.timestamp and result.audit_log[i - 1].timestamp:
                    dt = entry.timestamp - result.audit_log[i - 1].timestamp
                    elapsed = f" ({dt:.1f}s)"
                st.markdown(f"**{entry.stage}** -- {_escape_latex(entry.detail)}{elapsed}")
            if len(result.audit_log) >= 2:
                total = result.audit_log[-1].timestamp - result.audit_log[0].timestamp
                st.caption(f"Total processing time: {total:.1f}s")

    with st.expander("Raw JSON"):
        output = {
            "file_path": result.file_path,
            "stage_reached": result.stage_reached,
            "error": result.error,
        }
        if result.extraction:
            output["extraction"] = result.extraction.model_dump()
        if result.validation:
            output["validation"] = result.validation.model_dump()
        if result.approval:
            output["approval"] = result.approval.model_dump()
        if result.payment:
            output["payment"] = result.payment.model_dump()
        st.json(output)


def render_recent_invoices():
    rows = get_recent_invoices()
    if not rows:
        st.markdown(
            f"<div style='text-align:center;padding:32px;color:{C_TEXT_MUTED};"
            f"font-size:14px;'>No invoices processed yet.</div>",
            unsafe_allow_html=True,
        )
        return

    html = "<table class='inv-table'><thead><tr>"
    html += "<th>Vendor</th><th>Invoice ID</th><th>Amount</th><th>Status</th><th>Processed</th>"
    html += "</tr></thead><tbody>"

    for row in rows:
        vendor = row["vendor_name"] or "Unknown"
        initials = _get_initials(vendor)
        bg = _avatar_color(vendor)
        decision = row["decision"] or "unknown"
        color = STATUS_COLORS.get(decision, C_TEXT_MUTED)
        label = STATUS_LABELS.get(decision, decision.replace("_", " ").title())
        amount = f"${row['amount']:,.2f}" if row["amount"] else "--"
        processed = row["processed_on"][:10] if row["processed_on"] else "--"

        html += "<tr>"
        html += (
            f"<td><div class='vendor-cell'>"
            f"<div class='avatar' style='background:{bg};'>{initials}</div>"
            f"<div><div class='vendor-name'>{vendor}</div></div>"
            f"</div></td>"
        )
        html += f"<td>{row['invoice_number']}</td>"
        html += f"<td>{amount}</td>"
        html += (
            f"<td><span class='status-badge' "
            f"style='background:{color}15;color:{color};'>"
            f"{label}</span></td>"
        )
        html += f"<td style='color:{C_TEXT_MUTED};'>{processed}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def render_pipeline_overview():
    counts = get_pipeline_counts()
    if counts["total"] == 0:
        return

    st.markdown(
        f"<div class='pipeline-card'>"
        f"<div class='section-title'>Processing pipeline</div>"
        f"<div class='section-sub'>All-time invoice flow</div>"
        f"<div class='pipeline-stages'>"
        f"<div class='pipeline-stage'>"
        f"<div class='count'>{counts['total']}</div>"
        f"<div class='name'>Processed</div></div>"
        f"<div class='pipeline-arrow'>&#8250;</div>"
        f"<div class='pipeline-stage'>"
        f"<div class='count' style='color:{C_GREEN};'>{counts['paid']}</div>"
        f"<div class='name'>Paid</div></div>"
        f"<div class='pipeline-arrow'>&#8250;</div>"
        f"<div class='pipeline-stage'>"
        f"<div class='count' style='color:{C_AMBER};'>{counts['held']}</div>"
        f"<div class='name'>On hold</div></div>"
        f"<div class='pipeline-arrow'>&#8250;</div>"
        f"<div class='pipeline-stage'>"
        f"<div class='count' style='color:{C_RED};'>{counts['rejected']}</div>"
        f"<div class='name'>Rejected</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def main():
    ensure_db()

    # Sidebar
    with st.sidebar:
        st.markdown(
            f"<div class='brand'>"
            f"<div class='brand-icon'>A</div>"
            f"<div class='brand-name'>Auto<span>Invoice</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["Overview", "Process Invoices"],
            label_visibility="collapsed",
        )

    if page == "Overview":
        render_overview_page()
    else:
        render_process_page()


def render_overview_page():
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Invoice processing dashboard</div>", unsafe_allow_html=True)

    render_pipeline_overview()

    st.markdown("<div class='section-title'>Recent invoices</div>", unsafe_allow_html=True)
    count = get_pipeline_counts()["total"]
    st.markdown(
        f"<div class='section-sub'>{count} invoices processed</div>",
        unsafe_allow_html=True,
    )
    render_recent_invoices()


def render_process_page():
    st.markdown("<div class='section-title'>Process Invoices</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload and process invoice files</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload invoices",
        type=["txt", "json", "csv", "pdf", "xml", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            f"<div style='text-align:center;padding:48px 24px;color:{C_TEXT_MUTED};"
            f"font-size:14px;border:2px dashed {C_BORDER};border-radius:12px;"
            f"background:{C_BG_WARM};margin-top:8px;'>"
            f"Drop invoice files here or click to browse</div>",
            unsafe_allow_html=True,
        )
        return

    # Show queued files
    st.markdown("<div class='section-label'>Queued files</div>", unsafe_allow_html=True)
    for up_file in uploaded:
        ext = os.path.splitext(up_file.name)[1].upper().lstrip(".")
        initials = ext[:3]
        bg = _avatar_color(up_file.name)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"padding:10px 14px;background:{C_BG_CARD};border:1px solid {C_BORDER};"
            f"border-radius:8px;margin-bottom:6px;font-size:14px;color:{C_TEXT};'>"
            f"<div class='avatar' style='background:{bg};width:28px;height:28px;"
            f"font-size:10px;border-radius:6px;'>{initials}</div>"
            f"{up_file.name}"
            f"</div>",
            unsafe_allow_html=True,
        )

    if st.button("Process", type="primary", use_container_width=True):
        results: list[PipelineResult] = []
        progress = st.progress(0, text="Starting...")

        for i, up_file in enumerate(uploaded):
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, up_file.name)
            with open(tmp_path, "wb") as f:
                f.write(up_file.getvalue())

            progress.progress(
                i / len(uploaded),
                text=f"Processing {up_file.name} ({i + 1} of {len(uploaded)})",
            )
            results.append(process_invoice(tmp_path))

        progress.progress(1.0, text="Complete")
        st.session_state["results"] = results

    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    st.divider()

    # Summary metrics
    if len(results) > 1:
        approved = sum(1 for r in results if r.approval and r.approval.decision == "approved")
        rejected = sum(1 for r in results if r.approval and r.approval.decision == "rejected")
        held = sum(1 for r in results if r.approval and r.approval.decision == "needs_review")
        errors = sum(1 for r in results if r.error)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='value' style='color:{C_GREEN};'>{approved}</div>"
                f"<div class='label'>Approved</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='value' style='color:{C_RED};'>{rejected}</div>"
                f"<div class='label'>Rejected</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='value' style='color:{C_AMBER};'>{held}</div>"
                f"<div class='label'>Held</div></div>",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='value' style='color:{C_TEXT_MUTED};'>{errors}</div>"
                f"<div class='label'>Errors</div></div>",
                unsafe_allow_html=True,
            )
        st.divider()

    # Render each result
    for result in results:
        inv_label = (
            result.extraction.invoice.invoice_number
            if result.extraction else os.path.basename(result.file_path)
        )
        vendor = (
            result.extraction.invoice.vendor
            if result.extraction and result.extraction.invoice.vendor else ""
        )
        decision = result.approval.decision if result.approval else "error"
        if decision == "needs_review":
            decision = "held"

        header = inv_label
        if vendor:
            header += f" -- {vendor}"
        header += f" -- {decision.upper()}"

        if len(results) > 1:
            with st.expander(header, expanded=False):
                render_result(result)
        else:
            render_result(result)

    # Download
    if results:
        st.divider()
        jsonl = "\n".join(
            json.dumps({
                "source_file": os.path.basename(r.file_path),
                "invoice": r.extraction.invoice.model_dump() if r.extraction else None,
                "decision": r.approval.decision if r.approval else None,
                "risk_score": r.validation.risk_score if r.validation else None,
                "payment_status": r.payment.status if r.payment else None,
            })
            for r in results
        )
        st.download_button(
            "Download Results (JSONL)",
            data=jsonl,
            file_name="results.jsonl",
            mime="application/jsonl",
        )


if __name__ == "__main__":
    main()
