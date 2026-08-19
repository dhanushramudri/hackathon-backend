"""Chat with one SOW -- a strictly document-scoped Q&A surface for an RM or
governance-committee reviewer to ask things like "does this SOW cover the
contractor prioritisation dashboard?" without reading the whole document.

Accuracy posture: the ENTIRE document (chunked, never summarized/truncated
below a real page or section boundary) is sent on every turn -- no
retrieval-miss risk from a short document like a SOW. The model is
instructed to answer ONLY from that text and to say plainly when the
document doesn't address a question, never speculate. Every citation the
model returns is verified programmatically against the real extracted text
before being shown -- a citation whose quoted text doesn't actually appear
in the document (a hallucinated quote) is dropped, never displayed as if it
were real.

Citation locations are honest about what each file format actually gives us:
  - PDF (native, or a .docx real-rendered via LibreOffice -- see
    docx_conversion_service.py): a real page number (pypdf gives true
    per-page text off the actual rendered pages).
  - Word (.docx) when LibreOffice isn't available to render it: Word's raw
    file carries no reliable page-number data at all (pagination is a
    rendering-time concept, not stored data) -- so a docx citation falls
    back to an exact verbatim quote plus its position in reading order
    ("Part N of M"), never a fabricated page number.
"""
import io
import json
import re

from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ai import llm
from app.services.docx_conversion_service import DocxConversionError, get_or_convert_pdf
from app.services.project_sow_service import get_sow_file_path

DOCX_CHUNK_TARGET_CHARS = 1200


class SowChatError(Exception):
    pass


def _iter_block_items(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _docx_table_text(table: Table) -> str:
    lines = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _chunk_pdf(content: bytes) -> list[dict]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise SowChatError(f"Could not read this PDF: {exc}") from exc

    chunks = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append({"location": f"Page {i + 1}", "text": text})
    return chunks


def _chunk_docx(content: bytes) -> list[dict]:
    import docx
    try:
        doc = docx.Document(io.BytesIO(content))
    except Exception as exc:
        raise SowChatError(f"Could not read this Word document: {exc}") from exc

    raw_blocks: list[str] = []
    buffer: list[str] = []

    def flush_buffer():
        if buffer:
            raw_blocks.append("\n".join(buffer))
            buffer.clear()

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            buffer.append(text)
            if sum(len(b) for b in buffer) >= DOCX_CHUNK_TARGET_CHARS:
                flush_buffer()
        else:
            flush_buffer()
            table_text = _docx_table_text(block)
            if table_text:
                raw_blocks.append(table_text)
    flush_buffer()

    total = len(raw_blocks)
    return [{"location": f"Part {i + 1} of {total}", "text": b} for i, b in enumerate(raw_blocks) if b.strip()]


def _get_chunks(project_code: str, filename: str, content: bytes) -> list[dict]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "docx"):
        raise SowChatError(f"Unsupported file type '.{ext}' -- only PDF and Word SOWs can be chatted with.")

    if ext == "pdf":
        chunks = _chunk_pdf(content)
    else:
        # Prefer real page-numbered citations over the arbitrary "Part N of
        # M" fallback -- render the actual docx to a real PDF (same engine a
        # person opening it in an office suite would see) and chunk that by
        # real page, same as any native PDF. Only falls back to paragraph-
        # based chunking if LibreOffice isn't available on this machine.
        try:
            pdf_path = get_or_convert_pdf(project_code, filename)
            chunks = _chunk_pdf(pdf_path.read_bytes())
        except DocxConversionError:
            chunks = _chunk_docx(content)

    if not chunks:
        raise SowChatError("Could not extract any real text from this file -- it may be a scanned image without a real text layer.")
    return chunks


_SYSTEM_PROMPT_TEMPLATE = """You are answering questions about ONE specific, real Statement of Work (SOW) document for a Resource Manager or governance-committee reviewer. You must answer using ONLY the document content given below -- never speculate, infer beyond it, or use outside/general knowledge about the client, JMAN, or the industry.

If the document does not address the question, say so plainly (e.g. "This SOW does not mention that") rather than guessing or giving a vague non-answer.

For every factual claim in your answer, include a citation quoting the EXACT verbatim text from the document that supports it (copy it character-for-character, do not paraphrase the quote itself) along with which part of the document it came from.

The document, split into labeled parts:
---
{chunks_text}
---

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"answer": "<your answer, plain text>", "citations": [{{"quote": "<exact verbatim text from the document>", "location": "<the part label it came from, e.g. 'Page 2' or 'Part 3 of 9'>"}}]}}
If nothing in the document supports the answer (e.g. you had to say it doesn't mention the topic), return an empty citations list."""


def _build_system_prompt(chunks: list[dict]) -> str:
    chunks_text = "\n\n".join(f"[{c['location']}]\n{c['text']}" for c in chunks)
    return _SYSTEM_PROMPT_TEMPLATE.format(chunks_text=chunks_text)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _call_llm(system_prompt: str, message: str, history: list[dict]) -> dict:
    providers = llm.get_providers()
    if not providers:
        raise SowChatError("No AI provider is configured -- set GEMINI_API_KEY, ANTHROPIC_API_KEY, or the Azure OpenAI vars in the backend .env file.")

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    for provider in providers:
        try:
            turn = provider.generate_with_tools(messages, [], temperature=0.0, max_tokens=700)
        except Exception:
            continue
        content = (turn or {}).get("content")
        if not content:
            continue
        cleaned = _JSON_FENCE_RE.sub("", content.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed.get("answer"), str):
            return parsed

    raise SowChatError("The AI could not produce a valid answer -- try rephrasing the question.")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _verify_citations(citations: list, chunks: list[dict]) -> list[dict]:
    """Verify each citation's quote is real AND resolve its location to the
    chunk that actually contains it -- the LLM's own claimed location is
    never trusted, since it can genuinely quote real text while misremembering
    which page/part it came from (confirmed in practice: a verbatim quote
    correctly pulled from page 9 was labeled "Page 8"). Trusting that label
    would send the viewer to the wrong page, where the text can never be
    found to highlight.
    """
    normalized_chunks = [(c["location"], _normalize(c["text"])) for c in chunks]

    verified = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        quote = str(c.get("quote") or "").strip()
        if not quote:
            continue
        target = _normalize(quote)
        if not target:
            continue

        resolved_location = None
        for location, text in normalized_chunks:
            if target in text:
                resolved_location = location
                break
        if resolved_location is None:
            # The quote may straddle a page/part boundary -- check adjacent
            # chunk pairs joined together before giving up on it.
            for i in range(len(normalized_chunks) - 1):
                loc_a, text_a = normalized_chunks[i]
                _, text_b = normalized_chunks[i + 1]
                if target in f"{text_a} {text_b}":
                    resolved_location = loc_a
                    break
        if resolved_location is None:
            continue

        verified.append({"quote": quote, "location": resolved_location})
    return verified


def ask_sow_question(project_code: str, filename: str, message: str, history: list[dict]) -> dict:
    path = get_sow_file_path(project_code, filename)
    if path is None:
        raise SowChatError(f"No SOW file '{filename}' on record for project '{project_code}'.")
    content = path.read_bytes()

    chunks = _get_chunks(project_code, filename, content)

    system_prompt = _build_system_prompt(chunks)
    result = _call_llm(system_prompt, message, history)

    citations = _verify_citations(result.get("citations", []), chunks)
    return {"answer": result["answer"].strip(), "citations": citations}
