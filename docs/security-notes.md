# Security Notes

- Row Level Security (RLS) is enabled on all tenant tables.
- Policies enforce `user_id = auth.uid()` for select/insert/update/delete.
- Supabase service-role usage is server-side only; user_id scope is still required per write.
- Storage bucket is private. Access uses short-lived presigned URLs.
- Secrets are loaded from environment variables only.
- Logging uses salted hash values and excludes PII and prompts.
- Dedicated delete-account endpoint supports GDPR deletion workflows.
