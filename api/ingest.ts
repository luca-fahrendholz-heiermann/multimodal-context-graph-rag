import crypto from 'node:crypto';
import type { Request, Response } from 'express';
import { DummyEmbeddingProvider } from '../rag-core/embedding_provider';
import { ingestDocument } from '../rag-core/ingest';
import { logger } from '../rag-core/logging';
import { createServerSupabaseClient } from '../rag-core/supabase';

export async function ingestHandler(req: Request, res: Response): Promise<void> {
  const startedAt = Date.now();
  const userId = String(res.locals.userId ?? '');
  const requestId = String(res.locals.requestId ?? crypto.randomUUID());
  if (!userId) {
    res.status(401).json({ message: 'Unauthorized.' });
    return;
  }

  try {
    const supabase = createServerSupabaseClient();
    const result = await ingestDocument(
      supabase,
      new DummyEmbeddingProvider(),
      { userId, requestId },
      {
        rawText: String(req.body.rawText ?? ''),
        sourceType: String(req.body.sourceType ?? 'upload'),
        metadata: req.body.metadata,
      },
    );
    logger.info({ requestId, userId, action: 'ingest', status: 'success', latencyMs: Date.now() - startedAt });
    res.status(201).json(result);
  } catch {
    logger.info({ requestId, userId, action: 'ingest', status: 'error', latencyMs: Date.now() - startedAt });
    res.status(500).json({ message: 'Ingestion failed.' });
  }
}
