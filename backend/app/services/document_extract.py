"""Shared, bounded text extraction for article uploads."""
from __future__ import annotations

import io
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


ALLOWED_EXTS = {".md", ".markdown", ".txt", ".docx", ".pdf"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_CONTENT_CHARS = 2_000_000
MAX_PDF_PAGES = 200


def validate_document(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError("不支持该文件类型（支持 MD / TXT / DOCX / PDF）")
    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("文件超过 20MB")
    return ext


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _extract_docx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError as exc:
                raise ValueError("DOCX 中缺少正文内容") from exc
            if info.file_size > MAX_FILE_BYTES * 4:
                raise ValueError("DOCX 解压后的正文过大")
            root = ET.fromstring(archive.read(info))
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError("DOCX 文件损坏或格式无效") from exc

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []
    for paragraph in root.iter(f"{ns}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{ns}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{ns}tab":
                parts.append("\t")
            elif node.tag in {f"{ns}br", f"{ns}cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_pdf(content: bytes) -> str:
    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("暂不支持加密 PDF")
            if doc.page_count > MAX_PDF_PAGES:
                raise ValueError(f"PDF 超过 {MAX_PDF_PAGES} 页，请拆分后上传")
            pages = [page.get_text("text") for page in doc]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 文件损坏或无法读取") from exc
    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_article_text(filename: str, content: bytes) -> tuple[str, str]:
    ext = validate_document(filename, content)
    if ext in {".md", ".markdown", ".txt"}:
        text = _decode_text(content)
    elif ext == ".docx":
        text = _extract_docx(content)
    else:
        text = _extract_pdf(content)
    text = text.replace("\x00", "").strip()
    if not text and ext != ".pdf":
        raise ValueError("未能从文件中提取到文字")
    if len(text) > MAX_CONTENT_CHARS:
        raise ValueError("文档正文超过 200 万字，请拆分后上传")
    return ext, text


def infer_title(filename: str, text: str) -> str:
    for raw in text.splitlines()[:30]:
        line = raw.strip().lstrip("#").strip()
        if 2 <= len(line) <= 200:
            return line
    return Path(filename).stem[:200] or "未命名文章"
