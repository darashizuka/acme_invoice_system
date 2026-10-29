"""LangGraph workflow — orchestrates the 4-agent invoice processing pipeline.

Graph topology:
    START → ingest → validate ─┬─→ approve → pay → END
                                │
                                └─→ (self-correction) → ingest (retry)
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.ingestion import run_ingestion
from agents.validation import run_validation
from agents.approval import run_approval
from agents.payment import run_payment
from config import MAX_RETRIES
from models.schemas import (
    ApprovalDecision,
    ExtractionResult,
    PaymentResult,
    PipelineResult,
    ValidationResult,
)
from tools.parsers import parse_file

logger = logging.getLogger(__name__)


class InvoiceState(TypedDict, total=False):
    file_path: str
    raw_content: str
    file_format: str
    extraction: ExtractionResult | None
    validation: ValidationResult | None
    approval: ApprovalDecision | None
    payment: PaymentResult | None
    retry_count: int
    correction_feedback: str
    error: str | None
    stage: str
    audit_log: list[dict[str, Any]]


def _log_event(state: InvoiceState, stage: str, detail: str) -> list[dict]:
    log = list(state.get("audit_log", []))
    log.append({"stage": stage, "detail": detail, "timestamp": time.time()})
    return log


def node_parse(state: InvoiceState) -> dict:
    """Parse the raw invoice file."""
    file_path = state["file_path"]
    try:
        raw_content, fmt = parse_file(file_path)
        return {
            "raw_content": raw_content,
            "file_format": fmt,
            "stage": "parsed",
            "audit_log": _log_event(state, "parse", f"Parsed {fmt} file"),
        }
    except Exception as e:
        logger.error("Parse failed: %s", e)
        return {
            "error": f"Parse error: {e}",
            "stage": "error",
            "audit_log": _log_event(state, "parse", f"ERROR: {e}"),
        }


def node_ingest(state: InvoiceState) -> dict:
    """Run the ingestion agent to extract structured data."""
    try:
        feedback = state.get("correction_feedback")
        result = run_ingestion(
            raw_content=state["raw_content"],
            file_format=state["file_format"],
            feedback=feedback if feedback else None,
        )
        detail = (
            f"Extracted {result['extraction'].invoice.invoice_number} "
            f"(confidence={result['extraction'].confidence:.2f})"
        )
        if feedback:
            detail = f"RE-EXTRACTION: {detail}"
        return {
            "extraction": result["extraction"],
            "stage": "ingested",
            "audit_log": _log_event(state, "ingest", detail),
        }
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        return {
            "error": f"Ingestion error: {e}",
            "stage": "error",
            "audit_log": _log_event(state, "ingest", f"ERROR: {e}"),
        }


def node_validate(state: InvoiceState) -> dict:
    """Run the validation agent."""
    try:
        result = run_validation(state["extraction"])
        detail = (
            f"Risk={result['validation'].risk_score:.2f}, "
            f"passed={result['validation'].passed}, "
            f"issues={len(result['validation'].data_issues)}, "
            f"fraud_signals={len(result['validation'].fraud_signals)}"
        )
        updates: dict[str, Any] = {
            "validation": result["validation"],
            "stage": "validated",
            "audit_log": _log_event(state, "validate", detail),
        }
        if result["needs_correction"]:
            updates["correction_feedback"] = result["correction_feedback"]
        else:
            updates["correction_feedback"] = ""
        return updates
    except Exception as e:
        logger.error("Validation failed: %s", e)
        return {
            "error": f"Validation error: {e}",
            "stage": "error",
            "audit_log": _log_event(state, "validate", f"ERROR: {e}"),
        }


def node_approve(state: InvoiceState) -> dict:
    """Run the approval agent with reflection loop."""
    try:
        result = run_approval(state["extraction"], state["validation"])
        detail = (
            f"Decision={result['approval'].decision}, "
            f"risk_level={result['approval'].risk_level}"
        )
        return {
            "approval": result["approval"],
            "stage": "approved",
            "audit_log": _log_event(state, "approve", detail),
        }
    except Exception as e:
        logger.error("Approval failed: %s", e)
        return {
            "error": f"Approval error: {e}",
            "stage": "error",
            "audit_log": _log_event(state, "approve", f"ERROR: {e}"),
        }


def node_pay(state: InvoiceState) -> dict:
    """Run the payment agent."""
    try:
        result = run_payment(state["extraction"], state["approval"], state.get("validation"))
        detail = f"Status={result['payment'].status}"
        return {
            "payment": result["payment"],
            "stage": "completed",
            "audit_log": _log_event(state, "payment", detail),
        }
    except Exception as e:
        logger.error("Payment failed: %s", e)
        return {
            "error": f"Payment error: {e}",
            "stage": "error",
            "audit_log": _log_event(state, "payment", f"ERROR: {e}"),
        }


def route_after_parse(state: InvoiceState) -> str:
    if state.get("error"):
        return END
    return "ingest"


def route_after_ingest(state: InvoiceState) -> str:
    if state.get("error") or state.get("extraction") is None:
        return END
    return "validate"


def route_after_validate(state: InvoiceState) -> str:
    if state.get("error"):
        return END
    feedback = state.get("correction_feedback", "")
    retry_count = state.get("retry_count", 0)
    if feedback and retry_count < MAX_RETRIES:
        return "reingest"
    return "approve"


def route_after_approve(state: InvoiceState) -> str:
    if state.get("error") or state.get("approval") is None:
        return END
    return "pay"


def node_reingest(state: InvoiceState) -> dict:
    """Bump retry count then re-run ingestion with correction feedback."""
    new_count = state.get("retry_count", 0) + 1
    logger.info("Self-correction loop: retry %d/%d", new_count, MAX_RETRIES)
    result = node_ingest(state)
    result["retry_count"] = new_count
    return result


def build_graph() -> StateGraph:
    """Build and compile the invoice processing workflow graph."""
    graph = StateGraph(InvoiceState)

    graph.add_node("parse", node_parse)
    graph.add_node("ingest", node_ingest)
    graph.add_node("validate", node_validate)
    graph.add_node("reingest", node_reingest)
    graph.add_node("approve", node_approve)
    graph.add_node("pay", node_pay)

    graph.set_entry_point("parse")
    graph.add_conditional_edges("parse", route_after_parse, {"ingest": "ingest", END: END})
    graph.add_conditional_edges("ingest", route_after_ingest, {"validate": "validate", END: END})
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"reingest": "reingest", "approve": "approve", END: END},
    )
    graph.add_edge("reingest", "validate")
    graph.add_conditional_edges("approve", route_after_approve, {"pay": "pay", END: END})
    graph.add_edge("pay", END)

    return graph


def compile_graph():
    """Return a compiled, ready-to-invoke graph."""
    return build_graph().compile()


def process_invoice(file_path: str) -> PipelineResult:
    """Process a single invoice through the full pipeline. Returns PipelineResult."""
    app = compile_graph()

    initial_state: InvoiceState = {
        "file_path": file_path,
        "raw_content": "",
        "file_format": "",
        "extraction": None,
        "validation": None,
        "approval": None,
        "payment": None,
        "retry_count": 0,
        "correction_feedback": "",
        "error": None,
        "stage": "init",
        "audit_log": [],
    }

    final_state = app.invoke(initial_state)

    audit_log = final_state.get("audit_log", [])

    return PipelineResult(
        file_path=file_path,
        extraction=final_state.get("extraction"),
        validation=final_state.get("validation"),
        approval=final_state.get("approval"),
        payment=final_state.get("payment"),
        error=final_state.get("error"),
        stage_reached=final_state.get("stage", "unknown"),
        audit_log=audit_log,
    )
