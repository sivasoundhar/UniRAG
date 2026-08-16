"""
Smoke tests for app/loaders — real assert-based pytest versions of the
__main__ self-tests already in file_loader.py and ocr_loader.py.
"""

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.config.settings import settings
from app.loaders.file_loader import load_file
from app.loaders.ocr_loader import load_image


def _tesseract_available() -> bool:
    """settings.tesseract_cmd defaults to a Linux path (matches Docker/CI,
    where apt installs it there) — not necessarily present on a bare dev
    machine, so this test's dependency is checked rather than assumed."""
    return shutil.which(settings.tesseract_cmd) is not None or Path(settings.tesseract_cmd).exists()


def test_load_file_reads_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("UniRAG is a reusable AI knowledge core.\n", encoding="utf-8")

    docs = load_file(str(file_path))

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "sample.txt"
    assert "UniRAG" in docs[0].page_content


def test_load_file_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(str(tmp_path / "does_not_exist.txt"))


def test_load_file_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "sample.exe"
    bad_file.write_bytes(b"not a real document")

    with pytest.raises(ValueError):
        load_file(str(bad_file))


@pytest.mark.skipif(not _tesseract_available(), reason="tesseract binary not found on this machine")
def test_load_image_ocr_reads_text(tmp_path):
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (400, 60), color="white")
    ImageDraw.Draw(image).text((10, 10), "UniRAG OCR test", fill="black")
    image.save(image_path)

    docs = load_image(str(image_path))

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "sample.png"
    assert len(docs[0].page_content.strip()) > 0
