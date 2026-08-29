# Security Policy

## Reporting a vulnerability

Use the hosting platform's private security-reporting channel to contact the maintainers. Do not open
a public issue for an unpatched vulnerability. Include impact, reproduction steps, and a suggested fix
when possible, without attaching real user data or active credentials.

## Safe local operation

- Never commit `.config.local.json`, `.env` files, access keys, callback secrets, workspaces, or logs.
- Bind the gateway, WebSocket server, and callback server to `127.0.0.1` unless remote access is intentional.
- Configure `channels.director_callback.secret` before exposing callbacks outside the local machine.
- Set `tools.restrictToWorkspace=true` when filesystem tools should remain inside the workspace.
- Keep `tools.exec.enable=false` unless shell execution is required.
- Add trusted download domains explicitly; the default allow-list is empty.
- Use least-privilege, rotating credentials for optional S3-compatible storage.
- Review the privacy policy of every configured model provider before sending sensitive prompts or media.

If a secret has entered Git history, deleting the current file is not enough. Revoke or rotate the
credential immediately, then clean the repository history according to the project's release process.
