# VM-Controlled Google Colab Jobs

Kaya keeps the job record, retry policy, logs, and Colab CLI process on the VM. The browser submits or monitors jobs; closing the browser does not stop a running Celery task. Google account credentials are imported directly from the official Colab CLI token file; Kaya does not run a browser callback or OAuth client.

## One-time VM setup

1. On each Google account, sign in once with the official Colab CLI and copy its `~/.config/colab-cli/token.json` file contents.

2. Configure the VM environment:

   ```env
   GOOGLE_TOKEN_ENCRYPTION_KEY=...
   FRONTEND_URL=https://YOUR_DOMAIN
   COLAB_CLI_BIN=/absolute/path/to/colab
   ```

3. Install the pinned VM requirements. `google-colab-cli==0.5.11` is intentional: the published 0.6.0 wheel currently fails to import `ColabRuntime`.

4. Run Django, Redis, Celery, and the Next.js frontend as persistent system services.

## UI workflow

1. Open **Settings → Connections** and choose **Connect account**.
2. Enter the account email and paste its Colab CLI `token.json`. Kaya encrypts it, writes a 0600 vault copy, and verifies Drive with `about.get`.
3. Open **Console → Script Sandbox & Jobs**.
4. Select **Background Compute Job**, an authorized account, persistent session name, and accelerator.
5. Submit Python code. The VM queues it, creates or reuses the Colab session, executes the code, retries transient failures, and stores stdout/stderr in the job detail page.

For long training, the submitted code should periodically save checkpoints to Google Drive. A Colab-side runtime termination cannot be prevented; on retry the script must detect and resume its latest checkpoint.

## Operational checks

```bash
colab version
systemctl status kaya-api kaya-worker kaya-redis kaya-frontend
curl --fail https://YOUR_DOMAIN/api/v1/health/
```

The VM worker is configured for jobs up to six hours. Google still controls accelerator availability, quota, idle policies, and maximum runtime duration.
