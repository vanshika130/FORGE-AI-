"""Tests for document ingestion, chunking, and hybrid retrieval scoring."""

from __future__ import annotations

from pathlib import Path

from forgeai.ingest import ingest_documents, recursive_split_text
from forgeai.retrieval import EmbeddingBackend, HybridIndex


def test_recursive_split_respects_size_and_nonempty() -> None:
    text = (
        "# Section 1\n\n"
        + ("Shore staff receive 21 days of annual leave after 6 months. " * 12)
        + "\n\n# Section 2\n\n"
        + ("Sick leave is 10 working days per year. " * 10)
    )
    chunks = recursive_split_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 300 for c in chunks)


def test_ingest_handles_csv_encodings_and_empty_files(tmp_path: Path) -> None:
    csv_file = tmp_path / "parts.csv"
    csv_file.write_bytes(
        "part_number,description,unit_price_usd\nSP-01,Impeller — Bronze,120.50\n".encode("cp1252")
    )
    txt_file = tmp_path / "empty.txt"
    txt_file.write_text("", encoding="utf-8")

    md_file = tmp_path / "policy.md"
    md_file.write_text(
        "# Leave Policy\n\nFull-time staff get 21 working days of annual leave.",
        encoding="utf-8",
    )

    digest, chunks = ingest_documents(tmp_path, chunk_size=400, chunk_overlap=50)
    assert digest.has_csv is True
    assert len(digest.documents) == 3
    assert any(c.chunk_id == "policy.md#0" for c in chunks)
    assert any(c.is_csv_row and c.chunk_id == "parts.csv#0" for c in chunks)


def test_hybrid_retrieval_ranks_relevant_chunk_first(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "Full-time shore staff receive 21 working days of annual leave per calendar year.",
        encoding="utf-8",
    )
    (tmp_path / "conduct.md").write_text(
        "Employees may not accept supplier gifts exceeding 50 USD in value.",
        encoding="utf-8",
    )
    _, chunks = ingest_documents(tmp_path)
    embedder = EmbeddingBackend(force_hashed=True)
    index = HybridIndex(chunks, embedder=embedder)

    results = index.retrieve("How many annual leave days do shore staff get?", top_k=2, alpha=0.5)
    assert results
    assert results[0].chunk.doc == "leave.md"
    assert 0.0 <= results[0].dense_score <= 1.0
    assert results[0].score > 0.0
