import { describe, expect, it } from 'vitest';
import { assertTenantScope } from '../rag-core/supabase';

describe('assertTenantScope', () => {
  it('allows same-tenant operations', () => {
    expect(() => assertTenantScope({ userId: 'tenant-a', requestId: 'r1' }, 'tenant-a')).not.toThrow();
  });

  it('rejects cross-tenant operations', () => {
    expect(() => assertTenantScope({ userId: 'tenant-a', requestId: 'r1' }, 'tenant-b')).toThrow(
      'Cross-tenant access denied.',
    );
  });
});
