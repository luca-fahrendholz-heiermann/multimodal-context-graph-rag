import crypto from 'node:crypto';

export type LogPayload = {
  requestId: string;
  userId: string;
  action: string;
  status: 'success' | 'error';
  latencyMs: number;
};

function hashUserId(userId: string): string {
  const salt = process.env.LOG_SALT;
  if (!salt) {
    throw new Error('LOG_SALT must be configured.');
  }
  return crypto.createHash('sha256').update(`${userId}:${salt}`).digest('hex');
}

export const logger = {
  info(payload: LogPayload): void {
    const output = {
      requestId: payload.requestId,
      userHash: hashUserId(payload.userId),
      action: payload.action,
      status: payload.status,
      latencyMs: payload.latencyMs,
    };
    console.info(JSON.stringify(output));
  },
};
