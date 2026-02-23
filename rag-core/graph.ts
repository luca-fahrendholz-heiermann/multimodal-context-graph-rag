import type { SupabaseClient } from '@supabase/supabase-js';

export async function createGraphStub(
  supabase: SupabaseClient,
  userId: string,
  documentId: string,
  label: string,
): Promise<void> {
  const { data: node, error: nodeError } = await supabase
    .from('graph_nodes')
    .insert({ user_id: userId, label, type: 'document', metadata: { document_id: documentId } })
    .select('id')
    .single();

  if (nodeError) throw new Error('Failed to create graph node.');

  await supabase.from('graph_edges').insert({
    user_id: userId,
    from_node_id: node.id,
    to_node_id: node.id,
    relation: 'self',
    weight: 1,
    metadata: { document_id: documentId },
  });
}
