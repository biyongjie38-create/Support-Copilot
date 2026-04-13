from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Protocol

from app.models import KnowledgeDocument, SearchResult


class KnowledgeStore(Protocol):
    def ingest_documents(self, documents: list[dict[str, str]]) -> list[KnowledgeDocument]: ...
    def list_knowledge_documents(self) -> list[KnowledgeDocument]: ...
    def search_knowledge(self, query: str, limit: int = 5) -> list[SearchResult]: ...


def stable_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def tokenize_text(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+", lowered)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    chinese_bigrams = ["".join(chinese_chars[index : index + 2]) for index in range(max(len(chinese_chars) - 1, 0))]
    chinese_trigrams = ["".join(chinese_chars[index : index + 3]) for index in range(max(len(chinese_chars) - 2, 0))]
    return latin_tokens + chinese_chars + chinese_bigrams + chinese_trigrams


def text_embedding_literal(text: str, dimensions: int = 1536) -> str:
    vector = [0.0] * dimensions
    for token in tokenize_text(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
    magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
    normalized = [round(value / magnitude, 6) for value in vector]
    return "[" + ",".join(str(value) for value in normalized) + "]"


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
    query_tokens = set(tokenize_text(query))
    candidate_tokens = set(tokenize_text(candidate))
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
    exact_bonus = 0.25 if query.strip() and query.strip().lower() in candidate.lower() else 0.0
    keyword_score = min(1.0, overlap + exact_bonus)
    return round(min(1.0, keyword_score * 0.72 + max(vector_score, 0.0) * 0.28), 4)


class InMemoryKnowledgeStore:
    def __init__(self) -> None:
        self.documents: dict[str, KnowledgeDocument] = {}
        self.chunks: list[SearchResult] = []

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
        scored = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                content=chunk.content,
                score=score_text_relevance(query, f"{chunk.title}\n{chunk.content}", 0.0),
            )
            for chunk in self.chunks
        ]
        return [item for item in sorted(scored, key=lambda item: item.score, reverse=True) if item.score > 0][:limit]


class PgVectorStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

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
                        text_embedding_literal(f"{title}\n{chunk}"),
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
        query_embedding = text_embedding_literal(query)
        candidate_limit = max(limit * 8, 40)
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
        scored = [
            SearchResult(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                title=row[2],
                source=row[3],
                content=row[4],
                score=score_text_relevance(query, f"{row[2]}\n{row[4]}", float(row[5] or 0.0)),
            )
            for row in rows
        ]
        return [item for item in sorted(scored, key=lambda item: item.score, reverse=True) if item.score > 0][:limit]
