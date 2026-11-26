"""Hybrid dense + BM25 retrieval with disk embedding cache and offline fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Optional

import httpx
import numpy as np

from forgeai.schemas import ChunkRecord, RetrievalConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_text(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used for BM25 and lexical checks."""
    return _TOKEN_RE.findall(text.lower())


class SimpleBM25:
    """Pure-Python BM25Okapi fallback when rank_bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 1.0
        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in corpus:
            freqs: dict[str, int] = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.doc_freqs.append(freqs)
            for term in freqs:
                df[term] = df.get(term, 0) + 1
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or not query_tokens:
            return scores
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i]
            s = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                s += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-6)
            scores[i] = s
        return scores


class EmbeddingBackend:
    """Embedding provider: sentence-transformers -> Ollama -> deterministic hashed n-grams."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ollama_model: str = "nomic-embed-text",
        force_hashed: bool = False,
    ) -> None:
        self.model_name = model_name
        self.ollama_model = ollama_model
        self._st_model: Any = None
        self.backend_name = "hashed_ngrams"

        if force_hashed or os.getenv("FORGEAI_OFFLINE_EMBEDDINGS", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(model_name, local_files_only=True)
            self.backend_name = f"st:{model_name}"
            return
        except Exception:
            try:
                # Only attempt downloading from HuggingFace if reachable within 1s
                httpx.head("https://huggingface.co", timeout=1.0)
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(model_name)
                self.backend_name = f"st:{model_name}"
                return
            except Exception:
                self._st_model = None

        # Check Ollama embeddings endpoint quickly
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                if any(ollama_model in m for m in models):
                    self.backend_name = f"ollama:{ollama_model}"
                    return
        except Exception:
            pass

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        if self._st_model is not None:
            vecs = self._st_model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            return vecs.astype(np.float32)

        if self.backend_name.startswith("ollama:"):
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            rows: list[list[float]] = []
            try:
                for t in texts:
                    resp = httpx.post(
                        f"{ollama_url}/api/embeddings",
                        json={"model": self.ollama_model, "prompt": t},
                        timeout=15.0,
                    )
                    resp.raise_for_status()
                    rows.append(resp.json()["embedding"])
                arr = np.asarray(rows, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
            except Exception:
                pass

        return self._encode_hashed(texts, dim=384)

    @staticmethod
    def _encode_hashed(texts: list[str], dim: int = 384) -> np.ndarray:
        """Deterministic word + char-trigram feature hashing fallback."""
        mat = np.zeros((len(texts), dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = tokenize_text(text)
            features = list(tokens)
            for tok in tokens:
                if len(tok) >= 3:
                    for i in range(len(tok) - 2):
                        features.append(f"c:{tok[i : i + 3]}")
            for feat in features:
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                val = int.from_bytes(digest, "little")
                col = val % dim
                sign = 1.0 if ((val >> 8) & 1) == 0 else -1.0
                weight = 1.0 if not feat.startswith("c:") else 0.35
                mat[row_idx, col] += sign * weight
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


@dataclass
class RetrievedChunk:
    """A chunk plus its dense, BM25, and hybrid retrieval scores."""

    chunk: ChunkRecord
    score: float
    dense_score: float
    bm25_score: float


class HybridIndex:
    """In-process hybrid FAISS/NumPy + BM25 index over ChunkRecords."""

    def __init__(
        self,
        chunks: list[ChunkRecord],
        embedder: EmbeddingBackend | None = None,
        cache_dir: str | Path | None = None,
        doc_hash: str = "",
        chunk_params_key: str = "",
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder or EmbeddingBackend()
        self.cache_dir = Path(cache_dir or ".forgeai_cache/embeddings")
        self.embeddings = self._build_or_load_embeddings(doc_hash, chunk_params_key)
        self._faiss_index = self._build_faiss(self.embeddings)
        self._bm25 = self._build_bm25(chunks)

    def _build_or_load_embeddings(
        self, doc_hash: str, chunk_params_key: str
    ) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 384), dtype=np.float32)

        cache_file: Optional[Path] = None
        if doc_hash and chunk_params_key:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            safe_backend = re.sub(r"[^a-zA-Z0-9_-]", "_", self.embedder.backend_name)
            key = hashlib.sha256(
                f"{doc_hash}|{chunk_params_key}|{safe_backend}|{len(self.chunks)}".encode("utf-8")
            ).hexdigest()[:20]
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    arr = np.load(cache_file)
                    if arr.shape[0] == len(self.chunks):
                        return arr.astype(np.float32)
                except Exception:
                    pass

        arr = self.embedder.encode([c.text for c in self.chunks])
        if cache_file is not None:
            try:
                np.save(cache_file, arr)
            except Exception:
                pass
        return arr

    @staticmethod
    def _build_faiss(embeddings: np.ndarray) -> Any:
        if embeddings.shape[0] == 0:
            return None
        try:
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
            return index
        except Exception:
            return None

    @staticmethod
    def _build_bm25(chunks: list[ChunkRecord]) -> Any:
        tokenized = [tokenize_text(c.text) for c in chunks]
        if not tokenized:
            return None
        # For tiny corpora (< 5 chunks), standard BM25Okapi IDF becomes <= 0 for common terms;
        # SimpleBM25 uses positive +1 smoothed IDF so small-doc retrieval stays accurate.
        if len(tokenized) < 5:
            return SimpleBM25(tokenized)
        try:
            from rank_bm25 import BM25Okapi

            return BM25Okapi(tokenized)
        except Exception:
            return SimpleBM25(tokenized)

    def search_all_scores(
        self, query: str, alpha: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (hybrid_scores, dense_scores, raw_bm25_scores) aligned to self.chunks."""
        n = len(self.chunks)
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty

        q_vec = self.embedder.encode([query])
        if self._faiss_index is not None:
            # Retrieve all scores cleanly via matrix dot or FAISS search for top-N
            dense = (self.embeddings @ q_vec[0]).astype(np.float32)
        else:
            dense = (self.embeddings @ q_vec[0]).astype(np.float32)
        dense_norm = np.clip(dense, 0.0, 1.0)

        q_tokens = tokenize_text(query)
        if self._bm25 is not None and q_tokens:
            raw_bm25 = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
        else:
            raw_bm25 = np.zeros(n, dtype=np.float32)

        max_bm25 = float(np.max(raw_bm25)) if raw_bm25.size else 0.0
        bm25_norm = raw_bm25 / max_bm25 if max_bm25 > 0 else np.zeros_like(raw_bm25)

        alpha_c = max(0.0, min(1.0, float(alpha)))
        hybrid = alpha_c * dense_norm + (1.0 - alpha_c) * bm25_norm
        return hybrid, dense_norm, raw_bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        alpha: float = 0.5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return top_k chunks whose hybrid score meets min_score."""
        if not self.chunks:
            return []
        hybrid, dense_norm, raw_bm25 = self.search_all_scores(query, alpha=alpha)
        order = np.argsort(-hybrid)
        results: list[RetrievedChunk] = []
        for idx in order:
            sc = float(hybrid[idx])
            if sc < min_score:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(idx)],
                    score=sc,
                    dense_score=float(dense_norm[idx]),
                    bm25_score=float(raw_bm25[idx]),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def retrieve_with_config(
        self, query: str, cfg: RetrievalConfig
    ) -> list[RetrievedChunk]:
        return self.retrieve(
            query=query,
            top_k=cfg.top_k,
            alpha=cfg.hybrid_alpha,
            min_score=cfg.min_score,
        )

    def cosine_similarity_texts(self, text_a: str, text_b: str) -> float:
        vecs = self.embedder.encode([text_a, text_b])
        if vecs.shape[0] < 2:
            return 0.0
        return float(np.dot(vecs[0], vecs[1]))
