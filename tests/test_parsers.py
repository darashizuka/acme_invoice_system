"""Tests for the multi-format invoice parsers."""

import os
import json
import tempfile

import pytest

from tools.parsers import parse_file


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_parse_txt(tmp_dir):
    path = os.path.join(tmp_dir, "inv.txt")
    with open(path, "w") as f:
        f.write("Invoice 1001\nVendor: Acme\nTotal: $500")
    content, fmt = parse_file(path)
    assert fmt == "txt"
    assert "Invoice 1001" in content
    assert "Acme" in content


def test_parse_json(tmp_dir):
    path = os.path.join(tmp_dir, "inv.json")
    data = {"invoice_number": "INV-1001", "vendor": "Acme", "total": 500}
    with open(path, "w") as f:
        json.dump(data, f)
    content, fmt = parse_file(path)
    assert fmt == "json"
    assert "INV-1001" in content


def test_parse_csv(tmp_dir):
    path = os.path.join(tmp_dir, "inv.csv")
    with open(path, "w") as f:
        f.write("item,qty,price\nWidgetA,10,25.00\n")
    content, fmt = parse_file(path)
    assert fmt == "csv"
    assert "WidgetA" in content


def test_parse_xml(tmp_dir):
    path = os.path.join(tmp_dir, "inv.xml")
    with open(path, "w") as f:
        f.write("<invoice><number>INV-1001</number><vendor>Acme</vendor></invoice>")
    content, fmt = parse_file(path)
    assert fmt == "xml"
    assert "INV-1001" in content


def test_unsupported_format(tmp_dir):
    path = os.path.join(tmp_dir, "inv.docx")
    with open(path, "w") as f:
        f.write("test")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_file(path)


def test_image_format_returns_base64(tmp_dir):
    """Image files are base64-encoded for LLM vision extraction."""
    from PIL import Image
    from tools.parsers import has_images, extract_images

    path = os.path.join(tmp_dir, "inv.png")
    img = Image.new("RGB", (100, 100), color="white")
    img.save(path)

    content, fmt = parse_file(path)
    assert fmt == "image"
    assert has_images(content)
    images = extract_images(content)
    assert len(images) == 1
    assert len(images[0]) > 100  # base64 string
