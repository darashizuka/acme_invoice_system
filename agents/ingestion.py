"""Ingestion agent — extracts structured invoice data from raw content using LLM."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from models.schemas import ExtractionResult
from tools.parsers import has_images, extract_images

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert invoice data extraction agent. Extract structured data from the \
raw invoice content provided. The content may be messy, contain OCR artifacts, typos, \
inconsistent formatting, or missing fields.

RULES:
1. Normalize item names to their canonical form (e.g., "Widget A", "WIDGET-A", \
"widget_a" → "WidgetA"; "Gadget X", "GADGET-X" → "GadgetX").
2. Normalize dates to YYYY-MM-DD. If a date is clearly invalid (e.g., "yesterday", \
uses letter O instead of zero), note it as an issue and attempt to fix it.
3. If the invoice number lacks a standard prefix, add "INV-" (e.g., "1002" → "INV-1002").
4. Flag negative quantities, empty vendor names, null due dates, pressure language \
("URGENT", "pay immediately", "wire transfer preferred"), and any suspicious content.
5. For amounts, fix OCR errors (letter O → digit 0) and parse correctly.
6. Detect the currency — default to USD unless evidence of another currency.
7. Calculate confidence based on data quality: 1.0 for clean data, lower for messy/incomplete.
8. List ALL anomalies and issues you find in the raw content.

Respond with valid JSON matching the ExtractionResult schema."""

CORRECTION_PROMPT = """\
Your previous extraction had issues found during validation. Please re-extract \
the invoice data, correcting the following problems:

{feedback}

Original raw content:
{raw_content}

Respond with corrected JSON matching the ExtractionResult schema."""


def run_ingestion(
    raw_content: str,
    file_format: str,
    feedback: str | None = None,
) -> dict[str, Any]:
    """Extract structured invoice data from raw content.

    Returns a dict with keys: extraction, raw_content, file_format
    """
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ExtractionResult)

    if feedback:
        user_msg = CORRECTION_PROMPT.format(feedback=feedback, raw_content=raw_content)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    elif has_images(raw_content):
        messages = _build_vision_messages(raw_content)
    else:
        user_msg = (
            f"File format: {file_format}\n\n"
            f"Raw invoice content:\n```\n{raw_content}\n```"
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

    result: ExtractionResult = structured_llm.invoke(messages)
    logger.info(
        "Extraction complete: %s (confidence=%.2f, issues=%d)",
        result.invoice.invoice_number,
        result.confidence,
        len(result.issues),
    )
    return {
        "extraction": result,
        "raw_content": raw_content,
        "file_format": file_format,
    }


def _build_vision_messages(raw_content: str) -> list:
    """Build multimodal messages with inline base64 images for LLM vision extraction."""
    images = extract_images(raw_content)
    content_parts: list[dict] = [
        {"type": "text", "text": (
            "This invoice was received as a scanned document / image. "
            "Extract all structured data you can read from the image(s) below."
        )},
    ]
    for b64 in images:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content_parts),
    ]
