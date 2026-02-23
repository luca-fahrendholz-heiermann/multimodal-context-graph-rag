# Data Processing Flow

1. Browser uploads content to API using authenticated user context.
2. API strips PII before chunking and embedding.
3. Sanitized chunks are embedded via a provider adapter.
4. Data is stored in Supabase (EU region): documents, embeddings, graph nodes/edges.
5. Query flow uses tenant-scoped retrieval plus optional graph expansion.
6. Responses are returned without exposing hidden raw PII.
