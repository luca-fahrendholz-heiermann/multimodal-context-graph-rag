const EMAIL_REGEX = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;
const PHONE_REGEX = /\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}\b/g;
const IPV4_REGEX = /\b(?:\d{1,3}\.){3}\d{1,3}\b/g;
const IBAN_REGEX = /\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b/gi;
const FULL_NAME_REGEX = /\b([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)\b/g;

export type PiiScanResult = {
  filteredText: string;
  highRisk: boolean;
  replacements: Array<'email' | 'phone' | 'ip' | 'iban' | 'name'>;
};

export function stripPII(text: string): PiiScanResult {
  let filtered = text;
  const replacements: PiiScanResult['replacements'] = [];

  const replace = (regex: RegExp, marker: string, kind: PiiScanResult['replacements'][number]) => {
    const next = filtered.replace(regex, marker);
    if (next !== filtered) replacements.push(kind);
    filtered = next;
  };

  replace(EMAIL_REGEX, '[REDACTED_EMAIL]', 'email');
  replace(PHONE_REGEX, '[REDACTED_PHONE]', 'phone');
  replace(IPV4_REGEX, '[REDACTED_IP]', 'ip');
  replace(IBAN_REGEX, '[REDACTED_IBAN]', 'iban');
  replace(FULL_NAME_REGEX, '[REDACTED_NAME]', 'name');

  return {
    filteredText: filtered,
    highRisk: replacements.length > 0,
    replacements,
  };
}
