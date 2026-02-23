import crypto from 'node:crypto';
import type { Request, Response } from 'express';
import { logger } from '../rag-core/logging';
import { createServerSupabaseClient } from '../rag-core/supabase';

const PRIVATE_BUCKET = process.env.SUPABASE_PRIVATE_BUCKET ?? 'private-documents';

export async function deleteAccountHandler(_req: Request, res: Response): Promise<void> {
  const startedAt = Date.now();
  const userId = String(res.locals.userId ?? '');
  const requestId = String(res.locals.requestId ?? crypto.randomUUID());
  if (!userId) {
    res.status(401).json({ message: 'Unauthorized.' });
    return;
  }

  try {
    const supabase = createServerSupabaseClient();

    const { data: objects } = await supabase.storage.from(PRIVATE_BUCKET).list(userId);
    if (objects && objects.length > 0) {
      await supabase.storage
        .from(PRIVATE_BUCKET)
        .remove(objects.map((obj) => `${userId}/${obj.name}`));
    }

    await supabase.from('documents').delete().eq('user_id', userId);
    await supabase.from('graph_nodes').delete().eq('user_id', userId);

    logger.info({ requestId, userId, action: 'delete-account', status: 'success', latencyMs: Date.now() - startedAt });
    res.status(200).json({ deleted: true });
  } catch {
    logger.info({ requestId, userId, action: 'delete-account', status: 'error', latencyMs: Date.now() - startedAt });
    res.status(500).json({ message: 'Account deletion failed.' });
  }
}
