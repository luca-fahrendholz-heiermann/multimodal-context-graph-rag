import crypto from 'node:crypto';
import type { Request, Response } from 'express';
import { DummyEmbeddingProvider } from '../rag-core/embedding_provider';
import { logger } from '../rag-core/logging';
import { queryRag } from '../rag-core/query';
import { createServerSupabaseClient } from '../rag-core/supabase';

export async function queryHandler(req: Request, res: Response): Promise<void> {
  const startedAt = Date.now();
  const userId = String(res.locals.userId ?? '');
  const requestId = String(res.locals.requestId ?? crypto.randomUUID());
  if (!userId) {
    res.status(401).json({ message: 'Unauthorized.' });
    return;
  }

  try {
    const supabase = createServerSupabaseClient();
    const result = await queryRag(supabase, new DummyEmbeddingProvider(), { userId, requestId }, {
      query: String(req.body.query ?? ''),
      topK: Number(req.body.topK ?? 5),
    });

    logger.info({ requestId, userId, action: 'query', status: 'success', latencyMs: Date.now() - startedAt });
    res.status(200).json(result);
  } catch {
    logger.info({ requestId, userId, action: 'query', status: 'error', latencyMs: Date.now() - startedAt });
    res.status(500).json({ message: 'Query failed.' });
  }
}
