import { describe, expect, it } from 'vitest';
import { stripPII } from '../rag-core/pii_filter';

describe('stripPII', () => {
  it('redacts common PII patterns', () => {
    const input = 'Max Mustermann nutzt max@example.com mit +49 170 1234567 und 192.168.1.10';
    const result = stripPII(input);

    expect(result.filteredText).toContain('[REDACTED_EMAIL]');
    expect(result.filteredText).toContain('[REDACTED_PHONE]');
    expect(result.filteredText).toContain('[REDACTED_IP]');
    expect(result.highRisk).toBe(true);
  });
});
