create extension if not exists vector;

create table if not exists knowledge_documents (
  id uuid primary key,
  slug text,
  title text not null,
  source text not null default 'demo',
  content text not null,
  created_at timestamptz not null default now()
);

alter table knowledge_documents add column if not exists slug text;
alter table knowledge_documents add column if not exists source text not null default 'demo';
alter table knowledge_documents add column if not exists created_at timestamptz not null default now();

do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_name = 'knowledge_documents' and column_name = 'owner_id'
  ) then
    execute 'alter table knowledge_documents alter column owner_id drop not null';
  end if;
end $$;

update knowledge_documents
set slug = 'legacy-' || id::text
where slug is null;

create unique index if not exists idx_knowledge_documents_slug on knowledge_documents(slug);

create table if not exists knowledge_chunks (
  id uuid primary key,
  document_id uuid not null references knowledge_documents(id) on delete cascade,
  chunk_index integer not null default 0,
  title text not null default '',
  source text not null default 'demo',
  content text not null,
  embedding vector(1536) not null,
  created_at timestamptz not null default now()
);

alter table knowledge_chunks add column if not exists chunk_index integer not null default 0;
alter table knowledge_chunks add column if not exists title text not null default '';
alter table knowledge_chunks add column if not exists source text not null default 'demo';
alter table knowledge_chunks add column if not exists created_at timestamptz not null default now();

do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_name = 'knowledge_chunks' and column_name = 'owner_id'
  ) then
    execute 'alter table knowledge_chunks alter column owner_id drop not null';
  end if;
end $$;

create index if not exists idx_knowledge_chunks_document on knowledge_chunks(document_id, chunk_index);
create index if not exists idx_knowledge_chunks_embedding on knowledge_chunks using hnsw (embedding vector_cosine_ops);
