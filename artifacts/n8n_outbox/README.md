# n8n Outbox

This directory is a generated outbox for alert payloads when dry-run dispatch is enabled.

Current policy:
- Keep this folder present for runtime writes.
- Keep `.gitkeep` to preserve the folder in version control.
- Rotate historical payloads to `archive/generated/n8n_outbox/<date>/` during cleanup.
