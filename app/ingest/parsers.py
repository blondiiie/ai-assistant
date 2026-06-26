from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {"pdf": "PDF", "docx": "DOCX", "txt": "TXT", "md": "MD"}


@dataclass
class TextBlock:
    text: str
    page: int | None
    section: str | None


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _unfold_wikilink(match: re.Match[str]) -> str:
    inner = match.group(1)
    if "|" in inner:
        return inner.split("|", 1)[1].strip()
    return inner.strip()


def _strip_frontmatter(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                return "\n".join(lines[i + 1 :])
    return text


def parse_md(file_path: str) -> list[TextBlock]:
    title = Path(file_path).stem
    with open(file_path, encoding="utf-8") as f:
        raw = f.read()
    text = _strip_frontmatter(raw)
    text = _WIKILINK_RE.sub(_unfold_wikilink, text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)

    blocks: list[TextBlock] = []
    cur_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            blocks.append(TextBlock(text=f"{title}\n{body}", page=None, section=cur_section))
        buffer = []

    for line in text.split("\n"):
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            title_heading = heading.group(2).strip()
            cur_section = title_heading
            buffer.append(title_heading)
        else:
            buffer.append(line)
    flush()

    if not blocks:
        return []
    return blocks


def parse_pdf(file_path: str) -> list[TextBlock]:
    import pdfplumber

    blocks: list[TextBlock] = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(TextBlock(text=text, page=i, section=None))
    return blocks


def parse_docx(file_path: str) -> list[TextBlock]:
    import docx

    doc = docx.Document(file_path)
    blocks: list[TextBlock] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(
                TextBlock(
                    text="\n".join(buffer).strip(),
                    page=None,
                    section=current_section,
                )
            )
            buffer.clear()

    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        text = (para.text or "").strip()
        if not text:
            continue
        if style.startswith("heading") or style == "title":
            flush()
            current_section = text
        buffer.append(text)
    flush()
    return blocks


def parse_txt(file_path: str) -> list[TextBlock]:
    with open(file_path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    return [TextBlock(text=text, page=1, section=None)]


PARSERS = {"PDF": parse_pdf, "DOCX": parse_docx, "TXT": parse_txt, "MD": parse_md}


def parse(file_path: str, document_type: str) -> list[TextBlock]:
    parser = PARSERS.get(document_type)
    if parser is None:
        raise ValueError(f"Неподдерживаемый формат: {document_type}")
    blocks = parser(file_path)
    if not blocks:
        raise ValueError("Не удалось извлечь текст из документа (возможно, скан или пустой файл)")
    return blocks
