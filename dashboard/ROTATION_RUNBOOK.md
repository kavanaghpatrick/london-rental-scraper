# Secret Rotation Runbook — Neon Postgres (`POSTGRES_URL`)

> Status: **PREP ONLY.** The actual rotation is performed by **you (the user)** in the
> Neon console + GitHub + Vercel. This document is the exact, ordered checklist.
> Nothing in this repo stores the live password, so no code change is required to
> rotate — only the secret stores below.

## WHEN to run this (trigger)

Rotate **immediately before the rebuilt fresh-scrape dataset is synced to prod** —
i.e. as part of bringing prod back online after the data-loss rebuild. Sequence
(confirmed with serving; lead to OK):
1. Fresh scrape rebuilds the local dataset (#41).
2. **Rotate `POSTGRES_URL` (this runbook).**
3. Run the non-destructive UPSERT sync (`scripts/sync_sqlite_to_postgres.py --execute
   --i-have-rotated-the-secret`) to load the rebuilt data into the freshly-credentialed
   prod. (That sync is now UPSERT + takes a fail-closed `pg_dump` backup first, so even
   that load can't destroy anything.)

> NOTE: rotation was previously framed as gated on "PITR recovery sign-off." That
> recovery is **dead** (Neon Free tier, 6h window, expired — data permanently lost),
> so that trigger will never fire. Don't wait on it. Rotation is needed regardless —
> the current `POSTGRES_URL` is the very credential that allowed the unauthorized
> `--execute` that truncated prod.

## Why rotate

The live Neon database password is currently present in plaintext in a local,
git-ignored env file (`dashboard/.env.prod`) and is also stored as the GitHub
Actions secret `POSTGRES_URL` and as a Vercel project env var. It has been read
during this audit, lives in a `.env.prod` file on disk, AND is the credential that
permitted the prod-truncating `--execute` — so it must be treated as **exposed** and
rotated before prod is repopulated.

Redacted for reference (never paste full values into chat, tickets, or commits):

| Secret | Where | Redacted value |
|--------|-------|----------------|
| Neon DB password | `dashboard/.env.prod` line 3 (inside `POSTGRES_URL`) | `npg_…TQgL` |
| `VERCEL_OIDC_TOKEN` (prod) | `dashboard/.env.prod` line 21 | `eyJh…tH4Xg` (JWT — short-lived, see note) |
| `VERCEL_OIDC_TOKEN` (dev) | `dashboard/.env.local`, `../.env.local` | `eyJh…lvEA` / `eyJh…8Jyw` (JWT — short-lived) |

> The `VERCEL_OIDC_TOKEN` values are short-lived Vercel-issued JWTs (≈12h `exp`)
> and are already expired; they regenerate automatically on `vercel env pull`.
> They are NOT a manual rotation target, but do not commit them.

## Where `POSTGRES_URL` / the connection string is referenced (full inventory)

Grepped the whole repo (`scrapy_project/`). The **password itself** appears in
exactly one file. Everything else reads it from the environment by name.

| Location | How it's used | Action on rotation |
|----------|---------------|--------------------|
| `dashboard/.env.prod` (line 3) | **Contains the literal password** inside `POSTGRES_URL=...` | **Update file** (step 4) — or delete; `vercel env pull` regenerates it |
| GitHub Actions secret `POSTGRES_URL` | `.github/workflows/daily-scrape.yml` line 22 (`POSTGRES_URL: ${{ secrets.POSTGRES_URL }}`) | **Update GitHub secret** (step 3) |
| Vercel project env var `POSTGRES_URL` | Read at runtime by `@vercel/postgres` in `dashboard/src/lib/db.ts` and `dashboard/src/app/api/init-db/route.ts` | **Update Vercel env var** (step 2) |
| `dashboard/src/app/page.tsx` (line 90) | Only an **error-message string** ("Make sure POSTGRES_URL is set") — not a secret | No action |
| Python scripts (`train_model_postgres.py`, `property_scraper/pipelines_postgres.py`, `property_scraper/extensions/audit_logger_postgres.py`, `scripts/generate_negotiation_report.py`, `scripts/ocr_enrich.py`) | Read `os.environ['POSTGRES_URL']` — no literal value | No action (env-driven) |

There are **no hardcoded connection strings** in tracked source code.
`dashboard/.env.prod` and the `.env.local` files are correctly git-ignored
(`.gitignore` excludes `.env`, `.env.local`, `.env.prod`, `.env*.prod`,
`.env*.local`) and are **not** present in git history — verified during the audit.

---

## Rotation procedure (USER performs all steps)

Do these in order. Steps 1–4 are the rotation; step 5 verifies.

### 1. Rotate the password in the Neon console
1. Log in to <https://console.neon.tech>.
2. Open the project → branch → **Roles** (role `neondb_owner`).
3. Choose **Reset password** (generates a new `npg_…` password).
4. Copy the **new full connection string** Neon shows. It looks like:
   `postgresql://neondb_owner:<NEW_PASSWORD>@ep-fancy-union-adjjk0kb-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require`
   - Keep the **pooled** (`-pooler`) host, since that's what's in use today.
   - The old password (`npg_…TQgL`) is invalidated immediately on reset, so do
     steps 2–4 promptly to avoid downtime.

### 2. Update the Vercel env var (production runtime)
Either via dashboard or CLI.

**CLI (from `dashboard/`):**
```bash
cd dashboard
vercel env rm POSTGRES_URL production   # remove old
vercel env add POSTGRES_URL production  # paste the NEW connection string when prompted
```
Then redeploy so the new value takes effect:
```bash
vercel --prod
```
**Dashboard:** Vercel → project `dashboard` → Settings → Environment Variables →
edit `POSTGRES_URL` (Production) → paste new value → Save → Redeploy.

### 3. Update the GitHub Actions secret
The daily scrape (`.github/workflows/daily-scrape.yml`) reads `secrets.POSTGRES_URL`.
```bash
# from the repo root, using the gh CLI (already configured)
gh secret set POSTGRES_URL --body 'postgresql://neondb_owner:<NEW_PASSWORD>@ep-fancy-union-adjjk0kb-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require'
```
Or: GitHub repo → Settings → Secrets and variables → Actions → `POSTGRES_URL` → Update.

> Note: the literal `\n` currently at the end of the value in `.env.prod` is a
> Vercel-export artifact. Do **not** include a trailing newline/`\n` in the
> GitHub secret or the psycopg2 connection in CI may fail.

### 4. Update the local `dashboard/.env.prod` (and refresh dev env)
```bash
cd dashboard
vercel env pull .env.prod --environment=production   # regenerates the file from Vercel
# (also refreshes VERCEL_OIDC_TOKEN automatically)
```
Or hand-edit line 3 of `dashboard/.env.prod`, replacing the value of
`POSTGRES_URL` with the new connection string. **Do not commit this file**
(it is git-ignored — keep it that way).

### 5. Verify
```bash
# CI path — trigger the workflow's connection check
gh workflow run "Daily Property Scrape" -f spiders=rightmove -f full_mode=false
gh run watch   # confirm the "Check Postgres connection" step prints a count

# Runtime path — the dashboard should load data without the
# "Make sure POSTGRES_URL is set" message on the homepage.
```

---

## In-repo changes (already done by the security task — no rotation dependency)
- `dashboard/next.config.js` — added security headers (CSP/HSTS/etc.).
- `dashboard/src/app/api/init-db/route.ts` — now POST-only + requires
  `INIT_DB_TOKEN`. **New secret to provision** (see below).
- `dashboard/package.json` — Next bumped to `^16.2.9`.

## New secret to provision: `INIT_DB_TOKEN`
The schema-init endpoint is now disabled unless `INIT_DB_TOKEN` (≥16 chars) is set.
```bash
# generate a strong token
openssl rand -hex 24
# add to Vercel (production) so the endpoint can be used post-deploy
cd dashboard && vercel env add INIT_DB_TOKEN production
```
Invoke the endpoint after deploy:
```bash
curl -X POST https://<your-host>/api/init-db -H "Authorization: Bearer $INIT_DB_TOKEN"
```
If you never need runtime schema init, simply leave `INIT_DB_TOKEN` unset — the
endpoint stays disabled (returns 503).
