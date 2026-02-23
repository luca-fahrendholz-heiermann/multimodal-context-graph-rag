alter table documents enable row level security;
alter table embeddings enable row level security;
alter table graph_nodes enable row level security;
alter table graph_edges enable row level security;

drop policy if exists documents_select_own on documents;
drop policy if exists documents_insert_own on documents;
drop policy if exists documents_update_own on documents;
drop policy if exists documents_delete_own on documents;
create policy documents_select_own on documents for select using (user_id = auth.uid());
create policy documents_insert_own on documents for insert with check (user_id = auth.uid());
create policy documents_update_own on documents for update using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy documents_delete_own on documents for delete using (user_id = auth.uid());

drop policy if exists embeddings_select_own on embeddings;
drop policy if exists embeddings_insert_own on embeddings;
drop policy if exists embeddings_update_own on embeddings;
drop policy if exists embeddings_delete_own on embeddings;
create policy embeddings_select_own on embeddings for select using (user_id = auth.uid());
create policy embeddings_insert_own on embeddings for insert with check (user_id = auth.uid());
create policy embeddings_update_own on embeddings for update using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy embeddings_delete_own on embeddings for delete using (user_id = auth.uid());

drop policy if exists graph_nodes_select_own on graph_nodes;
drop policy if exists graph_nodes_insert_own on graph_nodes;
drop policy if exists graph_nodes_update_own on graph_nodes;
drop policy if exists graph_nodes_delete_own on graph_nodes;
create policy graph_nodes_select_own on graph_nodes for select using (user_id = auth.uid());
create policy graph_nodes_insert_own on graph_nodes for insert with check (user_id = auth.uid());
create policy graph_nodes_update_own on graph_nodes for update using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy graph_nodes_delete_own on graph_nodes for delete using (user_id = auth.uid());

drop policy if exists graph_edges_select_own on graph_edges;
drop policy if exists graph_edges_insert_own on graph_edges;
drop policy if exists graph_edges_update_own on graph_edges;
drop policy if exists graph_edges_delete_own on graph_edges;
create policy graph_edges_select_own on graph_edges for select using (user_id = auth.uid());
create policy graph_edges_insert_own on graph_edges for insert with check (user_id = auth.uid());
create policy graph_edges_update_own on graph_edges for update using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy graph_edges_delete_own on graph_edges for delete using (user_id = auth.uid());
