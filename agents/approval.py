"""Approval agent — three-tier approval with rule-based checks and LLM reflection loop."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import CFO_APPROVAL_THRESHOLD, VP_APPROVAL_THRESHOLD, get_llm
from models.schemas import (
    ApprovalDecision,
    ExtractionResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)

REASONING_PROMPT = """\
You are a VP of Finance reviewing an invoice for approval. You must decide whether \
to APPROVE, REJECT, or HOLD (needs_review) this invoice.

APPROVAL RULES:
1. Invoices with critical fraud signals or duplicate payments -> REJECT immediately.
2. Vendor not found in vendor master -> REJECT (unknown payee).
3. Vendor INACTIVE -> REJECT. Vendor ON HOLD -> HOLD for Vendor Management.
4. Arithmetic errors -> HOLD for manual review (could be minor rounding or material).
5. Missing required fields (vendor, invoice number, items, total) -> REJECT.
6. Foreign currency invoices -> verify USD equivalent is reasonable.
7. Risk score >= 0.6 -> lean toward REJECT or HOLD.
8. Clean invoices with risk score < 0.3 and no flags -> APPROVE.
9. When in doubt, HOLD — paying a bad invoice is the most expensive mistake.

IMPORTANT:
- Stock mismatches (requested qty > available stock) are a legitimate concern. \
  They may indicate over-billing or undelivered goods. Factor them into risk assessment.
- Pricing deviations > 10% from reference prices are a flag but not automatic rejection.

Provide detailed reasoning for your decision. Consider the business impact: \
a rejected legitimate invoice causes delays, but an approved fraudulent invoice \
causes direct financial loss.

Invoice data and validation results follow."""

REFLECTION_PROMPT = """\
You are a senior auditor reviewing the approval decision below. Critically evaluate \
the reasoning — look for:
1. Are there risks the approver missed?
2. Is the decision too lenient given the flags?
3. Would a reasonable VP sign off on this reasoning?
4. Is the risk level assessment accurate?
5. Were stock mismatches properly factored into risk assessment?

If the decision should change, say so explicitly and explain why. \
If the decision is sound, confirm it with a brief note on what makes it solid.

Previous decision and reasoning:
Decision: {decision}
Risk level: {risk_level}
Reasoning: {reasoning}
Flags: {flags}"""


def run_approval(
    extraction: ExtractionResult,
    validation: ValidationResult,
) -> dict[str, Any]:
    """Run three-tier approval: auto-approve small clean invoices, VP review medium,
    hold large for CFO.
    """
    invoice = extraction.invoice
    amount_usd = validation.usd_total or invoice.total

    # Tier 3: >$50K requires CFO — hold for human review
    if amount_usd > CFO_APPROVAL_THRESHOLD:
        logger.info(
            "Invoice %s ($%.2f) exceeds CFO threshold — holding for CFO + VP review",
            invoice.invoice_number, amount_usd,
        )
        decision = _vp_review(extraction, validation)

        if _should_override(decision, validation):
            decision = _apply_override(decision, validation)

        existing_flags = decision.flags if decision.flags else []
        if "cfo_required" not in existing_flags:
            existing_flags = existing_flags + ["cfo_required"]

        if decision.decision != "rejected":
            decision = ApprovalDecision(
                decision="needs_review",
                reasoning=decision.reasoning,
                reflection=(
                    f"Amount ${amount_usd:,.2f} exceeds "
                    f"${CFO_APPROVAL_THRESHOLD:,} CFO threshold. "
                    f"Holding for CFO co-approval. {decision.reflection}"
                ),
                risk_level=decision.risk_level,
                flags=existing_flags,
            )
        else:
            decision = ApprovalDecision(
                decision="rejected",
                reasoning=decision.reasoning,
                reflection=decision.reflection,
                risk_level=decision.risk_level,
                flags=existing_flags,
            )
        return {"approval": decision}

    # Tier 1: <=$10K with clean validation — auto-approve
    if (
        amount_usd <= VP_APPROVAL_THRESHOLD
        and validation.passed
        and validation.risk_score < 0.3
        and not validation.fraud_signals
        and not validation.is_duplicate
        and not validation.arithmetic_errors
        and (validation.vendor_check and validation.vendor_check.found_in_system)
        and (validation.vendor_check and validation.vendor_check.status == "active")
    ):
        logger.info(
            "Invoice %s ($%.2f) auto-approved — clean, under $%s threshold",
            invoice.invoice_number, amount_usd, f"{VP_APPROVAL_THRESHOLD:,}",
        )
        return {
            "approval": ApprovalDecision(
                decision="approved",
                reasoning=(
                    f"Auto-approved: amount ${amount_usd:,.2f} is under "
                    f"${VP_APPROVAL_THRESHOLD:,} threshold, validation passed with "
                    f"risk score {validation.risk_score:.2f}, no fraud signals, "
                    f"vendor is active, no arithmetic errors."
                ),
                reflection="Auto-approval — clean invoice below VP threshold.",
                risk_level="low",
                flags=[],
            )
        }

    # Tier 2: $10K-$50K or flagged small invoices — VP review with LLM
    decision = _vp_review(extraction, validation)

    if _should_override(decision, validation):
        decision = _apply_override(decision, validation)

    logger.info(
        "Approval complete: %s -> %s (risk=%s)",
        invoice.invoice_number,
        decision.decision,
        decision.risk_level,
    )
    return {"approval": decision}


def _vp_review(
    extraction: ExtractionResult,
    validation: ValidationResult,
) -> ApprovalDecision:
    """Run VP-level LLM review with reflection loop."""
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ApprovalDecision)

    context = _build_context(extraction, validation)

    initial_decision: ApprovalDecision = structured_llm.invoke([
        SystemMessage(content=REASONING_PROMPT),
        HumanMessage(content=context),
    ])

    reflection_input = REFLECTION_PROMPT.format(
        decision=initial_decision.decision,
        risk_level=initial_decision.risk_level,
        reasoning=initial_decision.reasoning,
        flags=initial_decision.flags,
    )

    reflection_context = (
        f"{reflection_input}\n\nOriginal invoice context:\n{context}"
    )

    final_decision: ApprovalDecision = structured_llm.invoke([
        SystemMessage(content=(
            "You are a senior auditor. After reflecting on the initial decision, "
            "provide the FINAL approval decision. If your reflection changes the "
            "decision, update it. Fill in the 'reflection' field with your critique."
        )),
        HumanMessage(content=reflection_context),
    ])

    return final_decision


def _build_context(extraction: ExtractionResult, validation: ValidationResult) -> str:
    inv = extraction.invoice
    amount_usd = validation.usd_total or inv.total

    lines = [
        f"Invoice: {inv.invoice_number}",
        f"Vendor: {inv.vendor}",
        f"Date: {inv.date} | Due: {inv.due_date}",
        f"Total: {inv.currency} {inv.total:,.2f}",
        f"USD equivalent: ${amount_usd:,.2f}",
        f"Items: {len(inv.items)}",
        f"Payment terms: {inv.payment_terms}",
        f"Notes: {inv.notes}",
        "",
        f"Extraction confidence: {extraction.confidence}",
        f"Extraction issues: {extraction.issues}",
        f"Anomalies: {extraction.raw_anomalies}",
        "",
        "--- Validation Results ---",
        f"Validation passed: {validation.passed}",
        f"Risk score: {validation.risk_score}",
        f"Summary: {validation.summary}",
        f"Is duplicate: {validation.is_duplicate}",
        "",
    ]

    if validation.vendor_check:
        vc = validation.vendor_check
        lines.extend([
            "Vendor check:",
            f"  Found in vendor master: {vc.found_in_system}",
            f"  Vendor ID: {vc.vendor_id}",
            f"  Status: {vc.status}",
            f"  Registered currency: {vc.currency}",
            f"  Issue: {vc.issue}",
            "",
        ])

    lines.append("Inventory checks:")
    for check in validation.inventory_checks:
        lines.append(
            f"  - {check.item_name}: found={check.found_in_inventory}, "
            f"stock={check.available_stock}, requested={check.requested_quantity}, "
            f"sufficient={check.stock_sufficient}, issue={check.issue}"
        )
    lines.append(f"\nData issues: {validation.data_issues}")
    lines.append(f"Arithmetic errors: {validation.arithmetic_errors}")
    lines.append(f"Fraud signals: {validation.fraud_signals}")
    return "\n".join(lines)


def _should_override(decision: ApprovalDecision, validation: ValidationResult) -> bool:
    """Only override on deterministic facts — not LLM judgment calls like risk score or fraud."""
    if decision.decision != "approved":
        return False
    if validation.is_duplicate:
        return True
    if validation.vendor_check and not validation.vendor_check.found_in_system:
        return True
    return False


def _apply_override(
    decision: ApprovalDecision, validation: ValidationResult
) -> ApprovalDecision:
    """Override an unsafe LLM approval on deterministic facts only."""
    reasons = []
    if validation.is_duplicate:
        reasons.append("duplicate invoice")
    if validation.vendor_check and not validation.vendor_check.found_in_system:
        reasons.append("vendor not in vendor master")

    # Duplicates are deterministic — reject outright, no human review needed
    if validation.is_duplicate:
        logger.warning(
            "Safety override: duplicate invoice — rejecting",
        )
        return ApprovalDecision(
            decision="rejected",
            reasoning=f"Duplicate invoice — already paid/processed. {decision.reasoning}",
            reflection=(
                f"SAFETY OVERRIDE: Duplicate detected. "
                f"Original reflection: {decision.reflection}"
            ),
            risk_level="critical",
            flags=decision.flags + ["safety_override", "duplicate_rejected"],
        )

    logger.warning(
        "Safety override: LLM approved despite %s — escalating to manual review",
        ", ".join(reasons),
    )
    return ApprovalDecision(
        decision="needs_review",
        reasoning=decision.reasoning,
        reflection=(
            f"SAFETY OVERRIDE: Original decision was 'approved' but "
            f"{', '.join(reasons)}. Escalating to manual review. "
            f"Original reflection: {decision.reflection}"
        ),
        risk_level="critical",
        flags=decision.flags + ["safety_override"],
    )
