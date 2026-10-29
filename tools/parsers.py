"""Multi-format invoice file parsers.

Each parser returns raw content as a string suitable for LLM extraction.
The LLM agent handles interpretation — parsers just extract text faithfully.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import xml.etree.ElementTree as ET


def parse_file(file_path: str) -> tuple[str, str]:
    """Parse an invoice file and return (content_string, format_type)."""
    ext = os.path.splitext(file_path)[1].lower()
    parsers = {
        ".txt": _parse_txt,
        ".json": _parse_json,
        ".csv": _parse_csv,
        ".pdf": _parse_pdf,
        ".xml": _parse_xml,
    }
    parser = parsers.get(ext)
    image_formats = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
    if ext in image_formats:
        return _parse_image(file_path), "image"
    if not parser:
        supported = ", ".join(sorted(parsers.keys()))
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {supported}. "
            f"Image files (png, jpg, tiff) require OCR preprocessing."
        )
    return parser(file_path), ext.lstrip(".")


def _parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_json(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


def _parse_csv(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    rows = []
    reader = csv.reader(io.StringIO(content))
    for row in reader:
        rows.append(" | ".join(row))
    return f"CSV Content:\n" + "\n".join(rows)


def _parse_pdf(file_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    cleaned = [cell or "" for cell in row]
                    text_parts.append(" | ".join(cleaned))

    if text_parts:
        return "\n".join(text_parts)

    # No text — render pages as images for LLM vision extraction
    images_b64 = _pdf_pages_to_base64(file_path)
    if images_b64:
        return _IMAGE_MARKER + json.dumps(images_b64)

    return "[PDF contained no extractable text]"


_IMAGE_MARKER = "[IMAGES_BASE64]"


def has_images(content: str) -> bool:
    """Check if parsed content contains base64 images for LLM vision."""
    return content.startswith(_IMAGE_MARKER)


def extract_images(content: str) -> list[str]:
    """Extract base64 image strings from parsed content."""
    if not content.startswith(_IMAGE_MARKER):
        return []
    return json.loads(content[len(_IMAGE_MARKER):])


def _pdf_pages_to_base64(file_path: str) -> list[str] | None:
    """Render PDF pages as PNG images and return as base64 strings."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(file_path)
        images = []
        for page in doc:
            bitmap = page.render(scale=2)
            pil_img = bitmap.to_pil()
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())
        doc.close()
        return images if images else None
    except Exception:
        return None


def _parse_image(file_path: str) -> str:
    """Read an image file and encode as base64 for LLM vision extraction."""
    from PIL import Image
    img = Image.open(file_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return _IMAGE_MARKER + json.dumps([b64])


def _parse_xml(file_path: str) -> str:
    tree = ET.parse(file_path)
    root = tree.getroot()
    lines = []
    _xml_to_text(root, lines, indent=0)
    return "\n".join(lines)


def _xml_to_text(element: ET.Element, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    tag = element.tag
    text = (element.text or "").strip()
    if text:
        lines.append(f"{prefix}{tag}: {text}")
    else:
        lines.append(f"{prefix}{tag}:")
    for child in element:
        _xml_to_text(child, lines, indent + 1)
    tail = (element.tail or "").strip()
    if tail:
        lines.append(f"{prefix}{tail}")
