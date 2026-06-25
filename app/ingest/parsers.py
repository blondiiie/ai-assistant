from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_EXTENSIONS = {"pdf": "PDF", "docx": "DOCX", "txt": "TXT"}


@dataclass
class TextBlock:
    text: str
    page: int | None
    section: str | None


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


PARSERS = {"PDF": parse_pdf, "DOCX": parse_docx, "TXT": parse_txt}


def parse(file_path: str, document_type: str) -> list[TextBlock]:
    parser = PARSERS.get(document_type)
    if parser is None:
        raise ValueError(f"Неподдерживаемый формат: {document_type}")
    blocks = parser(file_path)
    if not blocks:
        raise ValueError("Не удалось извлечь текст из документа (возможно, скан или пустой файл)")
    return blocks
