"""Document ingestion, parsing, recursive chunking, and digest generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd

from forgeai.schemas import ChunkRecord, DocumentDigest, DocumentInfo

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".txt", ".md", ".docx"}
MAX_CHUNKS_PER_DOC = 400
MAX_CSV_INDEX_ROWS = 300


def compute_docs_hash(docs_dir: str | Path) -> str:
    """Compute a deterministic hash over all supported files in docs_dir."""
    root = Path(docs_dir)
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            h.update(path.name.encode("utf-8"))
            h.update(str(path.stat().st_size).encode("utf-8"))
            with path.open("rb") as fh:
                while chunk := fh.read(65536):
                    h.update(chunk)
    return h.hexdigest()[:16]


def _read_text_with_fallback(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def _extract_pdf_pages(path: Path) -> tuple[list[tuple[int, str]], OptionalWarning]:
    pages: list[tuple[int, str]] = []
    warning: str | None = None
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append((idx, text))
    except ImportError:
        pages = []
    except Exception as exc:
        warning = f"pypdf failed on {path.name}: {exc}"

    total_chars = sum(len(t) for _, t in pages)
    if total_chars < 50:
        try:
            import pdfplumber

            plumber_pages: list[tuple[int, str]] = []
            with pdfplumber.open(str(path)) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    txt = (page.extract_text() or "").strip()
                    tables = page.extract_tables() or []
                    table_lines = []
                    for tbl in tables:
                        for row in tbl:
                            cells = [str(c).strip() if c is not None else "" for c in row]
                            if any(cells):
                                table_lines.append(" | ".join(cells))
                    combined = txt
                    if table_lines:
                        combined = f"{txt}\n" + "\n".join(table_lines)
                    plumber_pages.append((idx, combined.strip()))
            if sum(len(t) for _, t in plumber_pages) > total_chars:
                pages = plumber_pages
                total_chars = sum(len(t) for _, t in pages)
        except ImportError:
            if not pages:
                warning = (
                    f"Neither pypdf nor pdfplumber is installed to read {path.name}."
                )
        except Exception as exc:
            warning = f"pdfplumber failed on {path.name}: {exc}"

    if total_chars < 40 and warning is None:
        warning = (
            f"PDF '{path.name}' appears empty or scanned (extracted only {total_chars} chars)."
        )
    return pages, warning


OptionalWarning = str | None


def _extract_headings(text: str, limit: int = 8) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > 100:
            continue
        if re.match(r"^#{1,4}\s+.+", line):
            headings.append(re.sub(r"^#{1,4}\s+", "", line))
        elif re.match(r"^\d+(\.\d+)*[.)]\s+[A-Z].+", line):
            headings.append(line)
        elif line.isupper() and 4 <= len(line) <= 70:
            headings.append(line)
        if len(headings) >= limit:
            break
    return headings


def _make_three_line_summary(text: str, headings: list[str]) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    selected: list[str] = []
    for ln in lines:
        clean = re.sub(r"^#{1,4}\s+", "", ln)
        if clean not in selected:
            selected.append(clean[:180])
        if len(selected) >= 3:
            break
    if not selected and headings:
        selected = headings[:3]
    return "\n".join(selected) if selected else "(empty document)"


def recursive_split_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 80
) -> list[str]:
    """Split text recursively by headings, paragraphs, sentences, then characters."""
    text = text.strip()
    if not text:
        return []
    chunk_size = max(100, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 2))

    # Split into semantic units first using heading / paragraph boundaries.
    heading_blocks = re.split(r"(?=\n#{1,4}\s+|\n\d+\.\s+[A-Z])", text)
    units: list[str] = []
    for block in heading_blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= chunk_size:
            units.append(block)
            continue
        for para in re.split(r"\n\s*\n", block):
            para = para.strip()
            if not para:
                continue
            if len(para) <= chunk_size:
                units.append(para)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= chunk_size:
                    units.append(sent)
                else:
                    step = max(1, chunk_size - chunk_overlap)
                    for start in range(0, len(sent), step):
                        part = sent[start : start + chunk_size].strip()
                        if part:
                            units.append(part)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                tail = current[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
                combined = f"{tail} {unit}".strip() if tail else unit
                current = combined[:chunk_size] if len(combined) > chunk_size else combined
            else:
                chunks.append(unit[:chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def ingest_documents(
    docs_dir: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    source_documents: list[str] | None = None,
) -> tuple[DocumentDigest, list[ChunkRecord]]:
    """Parse all supported documents in docs_dir and return a digest plus stable chunks."""
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {root}")

    files = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if source_documents:
        allowed = set(source_documents)
        filtered = [p for p in files if p.name in allowed]
        if filtered:
            files = filtered

    doc_infos: list[DocumentInfo] = []
    all_chunks: list[ChunkRecord] = []
    warnings: list[str] = []
    has_csv = False

    for path in files:
        ext = path.suffix.lower()
        size_bytes = path.stat().st_size
        doc_warning: str | None = None

        if ext in {".csv", ".xlsx", ".xls"}:
            has_csv = True
            try:
                if ext == ".csv":
                    df = _read_csv_with_fallback(path)
                else:
                    df = pd.read_excel(path)
            except Exception as exc:
                msg = f"Failed to parse tabular file '{path.name}': {exc}"
                warnings.append(msg)
                doc_infos.append(
                    DocumentInfo(
                        file_name=path.name,
                        file_type=ext.lstrip("."),
                        size_bytes=size_bytes,
                        num_chars=0,
                        summary="(unreadable tabular file)",
                        warning=msg,
                    )
                )
                continue

            cols = [str(c) for c in df.columns.tolist()]
            sample_rows = df.head(3).fillna("").to_dict(orient="records")
            row_chunks: list[ChunkRecord] = []
            max_rows = min(len(df), MAX_CSV_INDEX_ROWS)
            if len(df) > MAX_CSV_INDEX_ROWS:
                doc_warning = (
                    f"CSV '{path.name}' has {len(df)} rows; indexed first {MAX_CSV_INDEX_ROWS} "
                    "row summaries for semantic search (use csv_lookup for full table queries)."
                )
                warnings.append(doc_warning)

            for r_idx in range(max_rows):
                row_dict = df.iloc[r_idx].fillna("").to_dict()
                summary_text = "; ".join(f"{k}: {v}" for k, v in row_dict.items())
                row_chunks.append(
                    ChunkRecord(
                        doc=path.name,
                        page=1,
                        chunk_id=f"{path.name}#{r_idx}",
                        text=summary_text,
                        is_csv_row=True,
                    )
                )
            all_chunks.extend(row_chunks)
            summary = (
                f"Tabular dataset ({len(df)} rows, {len(cols)} columns).\n"
                f"Columns: {', '.join(cols[:10])}\n"
                f"Use csv_lookup for exact filtering and aggregations."
            )
            doc_infos.append(
                DocumentInfo(
                    file_name=path.name,
                    file_type=ext.lstrip("."),
                    size_bytes=size_bytes,
                    num_chars=sum(len(c.text) for c in row_chunks),
                    num_chunks=len(row_chunks),
                    summary=summary,
                    top_headings=cols[:8],
                    csv_columns=cols,
                    csv_sample_rows=sample_rows,
                    warning=doc_warning,
                )
            )
            continue

        pages: list[tuple[int, str]] = []
        if ext == ".pdf":
            pages, doc_warning = _extract_pdf_pages(path)
        elif ext in {".txt", ".md"}:
            txt = _read_text_with_fallback(path)
            pages = [(1, txt)]
        elif ext == ".docx":
            try:
                import docx

                document = docx.Document(str(path))
                txt = "\n".join(p.text for p in document.paragraphs if p.text.strip())
                pages = [(1, txt)]
            except ImportError:
                doc_warning = f"python-docx not installed; skipping {path.name}"
            except Exception as exc:
                doc_warning = f"Failed to read docx '{path.name}': {exc}"

        if doc_warning:
            warnings.append(doc_warning)

        full_text = "\n\n".join(t for _, t in pages if t.strip()).strip()
        headings = _extract_headings(full_text)
        summary = _make_three_line_summary(full_text, headings)

        doc_chunks: list[ChunkRecord] = []
        chunk_idx = 0
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            parts = recursive_split_text(
                page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            for part in parts:
                if chunk_idx >= MAX_CHUNKS_PER_DOC:
                    cap_msg = (
                        f"Document '{path.name}' exceeded {MAX_CHUNKS_PER_DOC} chunks; truncated."
                    )
                    if cap_msg not in warnings:
                        warnings.append(cap_msg)
                        doc_warning = cap_msg
                    break
                doc_chunks.append(
                    ChunkRecord(
                        doc=path.name,
                        page=page_num,
                        chunk_id=f"{path.name}#{chunk_idx}",
                        text=part,
                        is_csv_row=False,
                    )
                )
                chunk_idx += 1
            if chunk_idx >= MAX_CHUNKS_PER_DOC:
                break

        all_chunks.extend(doc_chunks)
        doc_infos.append(
            DocumentInfo(
                file_name=path.name,
                file_type=ext.lstrip("."),
                size_bytes=size_bytes,
                num_chars=len(full_text),
                num_chunks=len(doc_chunks),
                summary=summary,
                top_headings=headings,
                warning=doc_warning,
            )
        )

    digest = DocumentDigest(documents=doc_infos, warnings=warnings, has_csv=has_csv)
    return digest, all_chunks


def format_digest_for_prompt(digest: DocumentDigest) -> str:
    """Format a DocumentDigest cleanly for LLM prompts."""
    lines: list[str] = []
    for doc in digest.documents:
        lines.append(
            f"- {doc.file_name} ({doc.file_type}, {doc.size_bytes} bytes, {doc.num_chunks} chunks)"
        )
        if doc.top_headings:
            lines.append(f"  Headings/Columns: {', '.join(doc.top_headings)}")
        lines.append("  Summary:")
        for s_line in doc.summary.splitlines():
            lines.append(f"    {s_line}")
        if doc.csv_sample_rows:
            lines.append(f"  Sample rows: {doc.csv_sample_rows[:2]}")
        if doc.warning:
            lines.append(f"  Warning: {doc.warning}")
    if digest.warnings:
        lines.append("Warnings: " + "; ".join(digest.warnings))
    return "\n".join(lines) if lines else "(No documents found)"
