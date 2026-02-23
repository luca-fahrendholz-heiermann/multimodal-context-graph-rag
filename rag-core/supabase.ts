import { createClient, type SupabaseClient } from '@supabase/supabase-js';

export type RequestContext = {
  userId: string;
  requestId: string;
};

export function createServerSupabaseClient(): SupabaseClient {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error('Supabase server credentials are not configured.');
  }
  return createClient(url, key, {
    auth: { persistSession: false },
  });
}

export function assertTenantScope(context: RequestContext, userId: string): void {
  if (context.userId !== userId) {
    throw new Error('Cross-tenant access denied.');
  }
}
