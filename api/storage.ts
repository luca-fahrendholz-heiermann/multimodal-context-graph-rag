import type { SupabaseClient } from '@supabase/supabase-js';

const PRIVATE_BUCKET = process.env.SUPABASE_PRIVATE_BUCKET ?? 'private-documents';

export async function createUploadUrl(
  supabase: SupabaseClient,
  userId: string,
  documentId: string,
  fileName: string,
): Promise<{ path: string; token: string }> {
  const path = `${userId}/${documentId}/${fileName}`;
  const { data, error } = await supabase.storage.from(PRIVATE_BUCKET).createSignedUploadUrl(path);
  if (error || !data) {
    throw new Error('Failed to create upload URL.');
  }
  return { path, token: data.token };
}

export async function createDownloadUrl(
  supabase: SupabaseClient,
  path: string,
  expiresIn = 60,
): Promise<string> {
  const { data, error } = await supabase.storage.from(PRIVATE_BUCKET).createSignedUrl(path, expiresIn);
  if (error || !data) {
    throw new Error('Failed to create download URL.');
  }
  return data.signedUrl;
}
