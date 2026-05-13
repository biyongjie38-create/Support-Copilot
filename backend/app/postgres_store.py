from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Protocol

from app.config import get_settings
from app.models import KnowledgeDocument, SearchResult


class KnowledgeStore(Protocol):
    def ingest_documents(self, documents: list[dict[str, str]]) -> list[KnowledgeDocument]: ...
    def list_knowledge_documents(self) -> list[KnowledgeDocument]: ...
    def search_knowledge(self, query: str, limit: int = 5) -> list[SearchResult]: ...


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_query(self, text: str) -> list[float]: ...


class LocalLexicalEmbeddingProvider:
    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return text_embedding_vector(text, dimensions=self.dimensions)


class LangChainEmbeddingProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None,
        dimensions: int = 1536,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.dimensions = dimensions
        self.embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        vector = self.embeddings.embed_query(text)
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Embedding model returned {len(vector)} dimensions, expected {self.dimensions}. "
                "Update LLM_EMBEDDING_DIMENSIONS and the pgvector schema together."
            )
        return [float(value) for value in vector]


def build_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    dimensions = settings.llm_embedding_dimensions
    api_key = settings.llm_api_key or ("ollama" if settings.llm_provider == "ollama" else None)
    if settings.llm_enable_calls and api_key:
        return LangChainEmbeddingProvider(
            model=settings.llm_embedding_model,
            api_key=api_key,
            base_url=settings.llm_base_url,
            dimensions=dimensions,
        )
    return LocalLexicalEmbeddingProvider(dimensions=dimensions)


def stable_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def tokenize_text(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+", lowered)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    chinese_bigrams = ["".join(chinese_chars[index : index + 2]) for index in range(max(len(chinese_chars) - 1, 0))]
    chinese_trigrams = ["".join(chinese_chars[index : index + 3]) for index in range(max(len(chinese_chars) - 2, 0))]
    return latin_tokens + chinese_chars + chinese_bigrams + chinese_trigrams


def alnum_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chinese_chars(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]", text.lower())


def chinese_ngrams(text: str, size: int) -> list[str]:
    chars = chinese_chars(text)
    return ["".join(chars[index : index + size]) for index in range(max(len(chars) - size + 1, 0))]


def overlap_ratio(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    if not query_set or not candidate_set:
        return 0.0
    return len(query_set & candidate_set) / len(query_set)


def text_embedding_vector(text: str, dimensions: int = 1536) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize_text(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
    magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
    return [round(value / magnitude, 6) for value in vector]


def text_embedding_literal(text: str, dimensions: int = 1536) -> str:
    vector = text_embedding_vector(text, dimensions=dimensions)
    return "[" + ",".join(str(value) for value in vector) + "]"


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(round(value, 8)) for value in vector) + "]"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return round(sum(a * b for a, b in zip(left, right)), 4)


def chunk_text(content: str, size: int = 900) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", content) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [content]:
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)
    return chunks or [content]


def score_text_relevance(query: str, candidate: str, vector_score: float = 0.0) -> float:
    query_alnum = alnum_tokens(query)
    candidate_alnum = alnum_tokens(candidate)
    alnum_score = overlap_ratio(query_alnum, candidate_alnum)
    bigram_score = overlap_ratio(chinese_ngrams(query, 2), chinese_ngrams(candidate, 2))
    trigram_score = overlap_ratio(chinese_ngrams(query, 3), chinese_ngrams(candidate, 3))
    char_score = overlap_ratio(chinese_chars(query), chinese_chars(candidate))
    cjk_score = max(trigram_score, bigram_score * 0.92, char_score * 0.45)
    lexical_score = max(
        alnum_score * 0.7 + cjk_score * 0.3,
        cjk_score,
        char_score * 0.35,
    )
    if query.strip() and query.strip().lower() in candidate.lower():
        lexical_score = min(1.0, lexical_score + 0.25)
    if any(token in {"api", "429"} for token in query_alnum) and {"api", "429"} <= set(candidate_alnum):
        lexical_score = min(1.0, lexical_score + 0.15)
    return round(min(1.0, lexical_score * 0.68 + max(vector_score, 0.0) * 0.32), 4)


def important_search_terms(query: str, max_terms: int = 8) -> list[str]:
    terms: list[str] = []
    for token in alnum_tokens(query):
        if len(token) >= 2 or token.isdigit():
            terms.append(token)
    terms.extend(token for token in chinese_ngrams(query, 3) if len(token) >= 3)
    terms.extend(token for token in chinese_ngrams(query, 2) if len(token) >= 2)
    deduped: list[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped[:max_terms]


def rerank_search_results(query: str, candidates: list[SearchResult], limit: int) -> list[SearchResult]:
    rescored = [
        SearchResult(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            title=candidate.title,
            source=candidate.source,
            content=candidate.content,
            score=score_text_relevance(query, f"{candidate.title}\n{candidate.content}", candidate.score),
        )
        for candidate in candidates
    ]
    return [item for item in sorted(rescored, key=lambda item: item.score, reverse=True) if item.score > 0][:limit]


class InMemoryKnowledgeStore:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.documents: dict[str, KnowledgeDocument] = {}
        self.chunks: list[SearchResult] = []
        self.embedding_provider = embedding_provider or build_embedding_provider()

    def ingest_documents(self, documents: list[dict[str, str]]) -> list[KnowledgeDocument]:
        ingested: list[KnowledgeDocument] = []
        for item in documents:
            document = KnowledgeDocument(
                id=stable_id(item["slug"]),
                slug=item["slug"],
                title=item["title"],
                source=item["source"],
                content=item["content"],
            )
            self.documents[document.id] = document
            self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document.id]
            for index, chunk in enumerate(chunk_text(document.content)):
                self.chunks.append(
                    SearchResult(
                        chunk_id=stable_id(f"{document.id}:{index}"),
                        document_id=document.id,
                        title=document.title,
                        source=document.source,
                        content=chunk,
                        score=1.0,
                    )
                )
            ingested.append(document)
        return ingested

    def list_knowledge_documents(self) -> list[KnowledgeDocument]:
        return sorted(self.documents.values(), key=lambda item: item.title)

    def search_knowledge(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_embedding = self.embedding_provider.embed_query(query)
        candidates = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                content=chunk.content,
                score=cosine_similarity(
                    query_embedding,
                    self.embedding_provider.embed_query(f"{chunk.title}\n{chunk.content}"),
                ),
            )
            for chunk in self.chunks
        ]
        return rerank_search_results(query, candidates, limit)


class PgVectorStore:
    def __init__(self, database_url: str, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.database_url = database_url
        self.embedding_provider = embedding_provider or build_embedding_provider()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def ingest_documents(self, documents: list[dict[str, str]]) -> list[KnowledgeDocument]:
        return [self.upsert_knowledge_document(**item) for item in documents]

    def upsert_knowledge_document(self, slug: str, title: str, source: str, content: str) -> KnowledgeDocument:
        document_id = stable_id(slug)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                insert into knowledge_documents (id, slug, title, source, content)
                values (%s, %s, %s, %s, %s)
                on conflict (slug) do update
                set title = excluded.title,
                    source = excluded.source,
                    content = excluded.content
                returning id, slug, title, source, content, created_at
                """,
                (document_id, slug, title, source, content),
            )
            row = cur.fetchone()
            cur.execute("delete from knowledge_chunks where document_id = %s", (row[0],))
            for index, chunk in enumerate(chunk_text(content)):
                cur.execute(
                    """
                    insert into knowledge_chunks (id, document_id, chunk_index, title, source, content, embedding)
                    values (%s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        stable_id(f"{row[0]}:{index}"),
                        row[0],
                        index,
                        title,
                        source,
                        chunk,
                        vector_literal(self.embedding_provider.embed_query(f"{title}\n{chunk}")),
                    ),
                )
        return KnowledgeDocument(
            id=str(row[0]),
            slug=row[1],
            title=row[2],
            source=row[3],
            content=row[4],
            created_at=row[5],
        )

    def list_knowledge_documents(self) -> list[KnowledgeDocument]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "select id, slug, title, source, content, created_at from knowledge_documents order by title"
            )
            rows = cur.fetchall()
        return [
            KnowledgeDocument(
                id=str(row[0]),
                slug=row[1],
                title=row[2],
                source=row[3],
                content=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def search_knowledge(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_embedding = vector_literal(self.embedding_provider.embed_query(query))
        candidate_limit = max(limit * 12, 60)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                select
                  c.id,
                  d.id,
                  d.title,
                  d.source,
                  c.content,
                  1 - (c.embedding <=> %s::vector) as vector_score
                from knowledge_chunks c
                join knowledge_documents d on d.id = c.document_id
                order by c.embedding <=> %s::vector
                limit %s
                """,
                (query_embedding, query_embedding, candidate_limit),
            )
            rows = cur.fetchall()
            lexical_rows = []
            terms = important_search_terms(query)
            if terms:
                conditions = []
                params: list[Any] = []
                for term in terms:
                    conditions.append("(c.content ilike %s or d.title ilike %s)")
                    like_term = f"%{term}%"
                    params.extend([like_term, like_term])
                cur.execute(
                    f"""
                    select
                      c.id,
                      d.id,
                      d.title,
                      d.source,
                      c.content,
                      0.0 as vector_score
                    from knowledge_chunks c
                    join knowledge_documents d on d.id = c.document_id
                    where {" or ".join(conditions)}
                    limit %s
                    """,
                    (*params, candidate_limit),
                )
                lexical_rows = cur.fetchall()
        candidates_by_chunk_id: dict[str, SearchResult] = {}
        for row in [*rows, *lexical_rows]:
            chunk_id = str(row[0])
            vector_score = float(row[5] or 0.0)
            existing = candidates_by_chunk_id.get(chunk_id)
            if existing is not None and existing.score >= vector_score:
                continue
            candidates_by_chunk_id[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                document_id=str(row[1]),
                title=row[2],
                source=row[3],
                content=row[4],
                score=vector_score,
            )
        return rerank_search_results(query, list(candidates_by_chunk_id.values()), limit)
