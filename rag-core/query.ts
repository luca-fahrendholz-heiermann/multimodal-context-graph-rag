import type { SupabaseClient } from '@supabase/supabase-js';
import type { EmbeddingProvider } from './embedding_provider';
import { stripPII } from './pii_filter';
import { assertTenantScope, type RequestContext } from './supabase';

export type QueryInput = {
  query: string;
  topK?: number;
};

export async function queryRag(
  supabase: SupabaseClient,
  embeddingProvider: EmbeddingProvider,
  context: RequestContext,
  input: QueryInput,
): Promise<{ answer: string; chunks: string[] }> {
  assertTenantScope(context, context.userId);

  const sanitized = stripPII(input.query).filteredText;
  const [queryVector] = await embeddingProvider.embed([sanitized]);

  const { data: chunks, error } = await supabase.rpc('match_embeddings', {
    query_embedding: queryVector,
    match_count: input.topK ?? 5,
    filter_user_id: context.userId,
  });

  if (error) throw new Error('Query retrieval failed.');

  const chunkTexts = (chunks ?? []).map((row: { content: string }) => row.content);
  const contextBlock = chunkTexts.join('\n---\n');

  const answer = [
    'System: Gib keine personenbezogenen Daten aus.',
    `Frage: ${sanitized}`,
    `Kontext: ${contextBlock}`,
    'Antwort (Stub): Relevante Informationen wurden datenschutzkonform verarbeitet.',
  ].join('\n');

  return { answer, chunks: chunkTexts };
}
