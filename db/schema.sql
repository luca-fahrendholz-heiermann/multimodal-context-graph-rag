create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  source_type text,
  content text,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);

create table if not exists embeddings (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  user_id uuid not null,
  chunk_index int not null,
  content text not null,
  vector vector(1536) not null,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists graph_nodes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  label text not null,
  type text,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists graph_edges (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  from_node_id uuid not null references graph_nodes(id) on delete cascade,
  to_node_id uuid not null references graph_nodes(id) on delete cascade,
  relation text,
  weight real,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz not null default now()
);
