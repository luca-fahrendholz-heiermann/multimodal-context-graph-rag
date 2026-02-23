import type { SupabaseClient } from '@supabase/supabase-js';
import { stripPII } from './pii_filter';
import type { EmbeddingProvider } from './embedding_provider';
import { createGraphStub } from './graph';
import { assertTenantScope, type RequestContext } from './supabase';

export type IngestInput = {
  rawText: string;
  sourceType: string;
  metadata?: Record<string, unknown>;
};

function chunkText(text: string, chunkSize = 1200, overlap = 100): string[] {
  if (!text.trim()) return [];
  const chunks: string[] = [];
  let index = 0;
  while (index < text.length) {
    const next = text.slice(index, index + chunkSize);
    chunks.push(next);
    index += chunkSize - overlap;
  }
  return chunks;
}

export async function ingestDocument(
  supabase: SupabaseClient,
  embeddingProvider: EmbeddingProvider,
  context: RequestContext,
  input: IngestInput,
): Promise<{ documentId: string; chunkCount: number }> {
  assertTenantScope(context, context.userId);

  const scan = stripPII(input.rawText);
  const storedContent = scan.highRisk ? scan.filteredText : input.rawText;
  const chunks = chunkText(scan.filteredText);
  const vectors = await embeddingProvider.embed(chunks);

  const { data: document, error: documentError } = await supabase
    .from('documents')
    .insert({
      user_id: context.userId,
      source_type: input.sourceType,
      content: storedContent,
      metadata: input.metadata ?? {},
    })
    .select('id')
    .single();

  if (documentError) throw new Error('Document ingestion failed.');

  const rows = chunks.map((chunk, chunkIndex) => ({
    user_id: context.userId,
    document_id: document.id,
    chunk_index: chunkIndex,
    content: chunk,
    vector: vectors[chunkIndex],
    metadata: { ...(input.metadata ?? {}), pii_replacements: scan.replacements },
  }));

  if (rows.length > 0) {
    const { error: embeddingsError } = await supabase.from('embeddings').insert(rows);
    if (embeddingsError) throw new Error('Embedding persistence failed.');
  }

  await createGraphStub(supabase, context.userId, document.id, input.sourceType);

  return { documentId: document.id, chunkCount: rows.length };
}
