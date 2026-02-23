# Retention Policy

- Application logs are retained for a maximum of 30 days.
- Logs contain only request metadata (`requestId`, `userHash`, action, status, latency).
- Prompts, raw document text, and direct user identifiers are never written to logs.
- On account deletion, user-scoped documents, embeddings, graph data, and private storage artifacts are deleted immediately.
