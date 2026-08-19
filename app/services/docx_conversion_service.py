"""Converts a real, already-uploaded .docx SOW to a real PDF using a locally
installed LibreOffice (`soffice --headless --convert-to pdf`) -- an actual
office-document layout engine, not a text-extraction approximation, so the
result has the real fonts, colors, images, tables, and pagination the
original author saw. The result is cached to disk (re-converted only if the
source file changes), so this only costs real time once per uploaded file.

Why this exists: browsers have no native way to render .docx (unlike PDF,
where every browser ships a built-in viewer), so a pixel-perfect preview
needs a real rendering engine somewhere. This keeps that engine local
(no public hosting, no cloud service, no data leaving the machine) --
important since a SOW carries real client names and real financial terms.

If LibreOffice isn't installed/working (e.g. a different machine that never
ran the setup step), conversion raises DocxConversionError -- callers are
expected to fall back to a lower-fidelity preview rather than break the
feature entirely, since that failure mode is a real, expected possibility.
"""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.core.config import APP_STATE_DIR
from app.services.project_sow_service import get_sow_file_path

CONVERTED_PDF_DIR = APP_STATE_DIR / "sow_converted_pdfs"
CONVERTED_PDF_DIR.mkdir(exist_ok=True)

# LibreOffice's own install location isn't always on PATH after a fresh
# winget/msi install within the same shell session -- check the standard
# Program Files location too rather than relying purely on PATH resolution.
_SOFFICE_CANDIDATES = [
    "soffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]

CONVERSION_TIMEOUT_SECONDS = 60


class DocxConversionError(Exception):
    pass


def _find_soffice() -> str:
    for candidate in _SOFFICE_CANDIDATES:
        if candidate == "soffice":
            if shutil.which("soffice"):
                return "soffice"
        elif Path(candidate).exists():
            return candidate
    raise DocxConversionError("LibreOffice (soffice) isn't installed -- pixel-perfect docx preview isn't available on this machine.")


def _cache_path(project_code: str, filename: str) -> Path:
    key = hashlib.md5(f"{project_code}::{filename}".encode()).hexdigest()
    return CONVERTED_PDF_DIR / f"{key}.pdf"


def get_or_convert_pdf(project_code: str, filename: str) -> Path:
    """Returns a real PDF for this SOW -- the original file untouched if it's
    already a PDF, or a freshly-rendered (and cached) LibreOffice conversion
    if it's a .docx. Raises DocxConversionError if conversion is needed but
    unavailable/fails."""
    source_path = get_sow_file_path(project_code, filename)
    if source_path is None:
        raise DocxConversionError(f"No SOW file '{filename}' on record for project '{project_code}'.")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return source_path

    if ext != "docx":
        raise DocxConversionError(f"Unsupported file type '.{ext}' for PDF preview.")

    cached = _cache_path(project_code, filename)
    if cached.exists() and cached.stat().st_mtime >= source_path.stat().st_mtime:
        return cached

    # Re-created on every call, not just once at import -- this directory is
    # cheap to lose (a stale-cache cleanup, a disk cleanup tool, anything
    # deleting it out from under an already-running server process) and the
    # only symptom otherwise is copy2() below failing with a bare WinError 3
    # that gives no hint the cache directory itself is the thing missing.
    CONVERTED_PDF_DIR.mkdir(parents=True, exist_ok=True)

    soffice = _find_soffice()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            subprocess.run(
                [soffice, "--headless", "--norestore", "--convert-to", "pdf", "--outdir", tmp_dir, str(source_path)],
                check=True, capture_output=True, timeout=CONVERSION_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise DocxConversionError("LibreOffice (soffice) isn't installed -- pixel-perfect docx preview isn't available on this machine.") from exc
        except subprocess.TimeoutExpired as exc:
            raise DocxConversionError("Converting this document to PDF timed out.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode(errors="replace")
            raise DocxConversionError(f"LibreOffice could not convert this document: {detail[:300]}") from exc

        produced = Path(tmp_dir) / f"{source_path.stem}.pdf"
        if not produced.exists():
            raise DocxConversionError("LibreOffice did not produce a PDF for this document.")
        shutil.copy2(produced, cached)

    return cached
