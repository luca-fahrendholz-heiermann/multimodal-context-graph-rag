create index if not exists idx_documents_user_id on documents(user_id);
create index if not exists idx_embeddings_user_id on embeddings(user_id);
create index if not exists idx_embeddings_document_id on embeddings(document_id);
create index if not exists idx_graph_nodes_user_id on graph_nodes(user_id);
create index if not exists idx_graph_edges_user_id on graph_edges(user_id);
create index if not exists idx_graph_edges_from_node_id on graph_edges(from_node_id);
create index if not exists idx_graph_edges_to_node_id on graph_edges(to_node_id);

-- Choose one index type based on your Postgres setup.
create index if not exists idx_embeddings_vector_ivfflat
  on embeddings using ivfflat (vector vector_cosine_ops) with (lists = 100);
