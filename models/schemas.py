from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    item_name: str = Field(description="Normalized product name, e.g. 'WidgetA'")
    quantity: int = Field(description="Quantity ordered")
    unit_price: float = Field(description="Price per unit in the invoice currency")
    line_total: float | None = Field(default=None, description="Line total if provided")
    notes: str | None = None


class InvoiceData(BaseModel):
    invoice_number: str = Field(description="Invoice identifier, e.g. 'INV-1001'")
    vendor: str = Field(description="Vendor / supplier name")
    date: str | None = Field(default=None, description="Invoice date (YYYY-MM-DD)")
    due_date: str | None = Field(default=None, description="Due date (YYYY-MM-DD or null)")
    items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float = Field(description="Grand total amount")
    currency: str = Field(default="USD")
    payment_terms: str | None = None
    notes: str | None = None


class ExtractionResult(BaseModel):
    invoice: InvoiceData
    confidence: float = Field(ge=0, le=1, description="Extraction confidence 0-1")
    issues: list[str] = Field(default_factory=list, description="Issues found during extraction")
    raw_anomalies: list[str] = Field(default_factory=list, description="Anomalies in raw content")


class InventoryCheck(BaseModel):
    item_name: str
    found_in_inventory: bool
    available_stock: int | None = None
    requested_quantity: int
    stock_sufficient: bool | None = None
    issue: str | None = None


class VendorCheck(BaseModel):
    vendor_name: str
    found_in_system: bool
    vendor_id: str | None = None
    status: str | None = None
    currency: str | None = None
    issue: str | None = None


class ValidationResult(BaseModel):
    passed: bool
    vendor_check: VendorCheck | None = None
    inventory_checks: list[InventoryCheck] = Field(default_factory=list)
    data_issues: list[str] = Field(default_factory=list)
    fraud_signals: list[str] = Field(default_factory=list)
    arithmetic_errors: list[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0, le=1, description="Overall risk score 0-1")
    summary: str = ""
    is_duplicate: bool = False
    usd_total: float | None = None


class ApprovalDecision(BaseModel):
    decision: Literal["approved", "rejected", "needs_review"] = Field(
        description="Final decision"
    )
    reasoning: str = Field(description="Detailed reasoning for the decision")
    reflection: str = Field(description="Self-critique of the reasoning")
    risk_level: Literal["low", "medium", "high", "critical"] = "low"
    flags: list[str] = Field(default_factory=list)


class PaymentResult(BaseModel):
    status: Literal["paid", "rejected", "held"]
    vendor: str
    amount: float
    currency: str = "USD"
    reason: str | None = None


class AuditEntry(BaseModel):
    stage: str
    detail: str
    timestamp: float = 0.0


class PipelineResult(BaseModel):
    file_path: str
    extraction: ExtractionResult | None = None
    validation: ValidationResult | None = None
    approval: ApprovalDecision | None = None
    payment: PaymentResult | None = None
    error: str | None = None
    stage_reached: str = "init"
    audit_log: list[AuditEntry] = Field(default_factory=list)
