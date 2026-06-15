# Security Report — Dashboard Hardening (Task #10)

Scope (owned files): `dashboard/src/app/api/init-db/route.ts`,
`dashboard/next.config.js`, `dashboard/package.json`, plus new runbooks.
Not touched: `report/*`, `api/similar`, `api/running`, `lib/db.ts` (owned by serving #8).

All four runtime/config changes were verified: `next build` succeeds on Next
16.2.9, and a running `next start` server was curl-tested for headers + auth.

> Schema-parity addendum (requested by team-lead per DATA_LAYER_CONTRACT §3):
> `init-db/route.ts` now also defines `listings.canonical_id INTEGER` and
> `price_history.price_pw INTEGER` (in both the base CREATE and the idempotent
> ALTER migration). Types reconciled with serving's `sync_sqlite_to_postgres.py`
> (both use INTEGER, not TEXT) and verified live in Neon via
> `information_schema.columns` → both report `integer`. Build re-verified green.

---

## Findings (severity-ranked, before → after)

### 1. CRITICAL — Unauthenticated, public schema-mutating endpoint (`/api/init-db`)
- **Before:** `GET /api/init-db` was public and unauthenticated, and executed
  `CREATE TABLE` / `ALTER TABLE` / `CREATE INDEX` against production Postgres.
  Any visitor, crawler, or link-prefetch could trigger DDL on the live DB. A GET
  that mutates state is also CSRF-prone and cacheable.
- **After:** Rewritten to **POST-only** with a **fail-closed token guard**:
  - Requires `INIT_DB_TOKEN` (≥16 chars) in env; if unset → `503` (disabled).
  - Requires `Authorization: Bearer <token>` or `x-init-token: <token>`;
    wrong/missing → `401`. Constant-time comparison (no timing leak).
  - `GET` now returns `405 Method Not Allowed`.
  - `dynamic = 'force-dynamic'`, `runtime = 'nodejs'` so it's never cached/prerendered.
- **Verified:** GET→405; POST no-token→503; POST wrong-token (token set)→401;
  POST correct-token→200 (`{"success":true,...}`).

### 2. HIGH — Out-of-date Next.js with 20+ known CVEs (`next@16.0.10`)
- **Before:** `next@^16.0.10`. Affected by multiple **High** advisories incl.
  middleware/proxy bypass (CVE-2026-44573/44574/44575/45109), SSRF via WebSocket
  upgrades (CVE-2026-44578), and DoS via Cache Components / Server Components
  (CVE-2026-44579, GHSA-8h8q-6873-q5fj, GHSA-q4gf-8mx6-v5v3).
- **After:** Bumped to **`next@^16.2.9`** (current `latest`), which patches the
  above. Lockfile + `node_modules` updated; `next build` passes.
- **Verified:** installed version is `16.2.9`; production build green.

### 3. HIGH — Missing HTTP security headers (empty `next.config.js`)
- **Before:** `next.config.js` was `const nextConfig = {}` — no CSP, no
  anti-clickjacking, no HSTS, no MIME-sniff protection; `X-Powered-By: Next.js`
  was leaked.
- **After:** `headers()` applies to every route:
  - `Content-Security-Policy` (default-src 'self', `frame-ancestors 'none'`,
    `object-src 'none'`, `base-uri`/`form-action` 'self',
    `upgrade-insecure-requests`; `connect-src` limited to self + `*.vercel.app` +
    `*.neon.tech`). `'unsafe-inline'`/`'unsafe-eval'` retained for script/style
    because Next App Router + Recharts require inline.
  - `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
    `Referrer-Policy: strict-origin-when-cross-origin`,
    `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`,
    `Permissions-Policy: camera=(), microphone=(), geolocation=(), browsing-topics=()`,
    `X-DNS-Prefetch-Control: off`, and `poweredByHeader: false`.
- **Verified:** all headers present on `/` via `curl -I`; `X-Powered-By` gone.

### 4. HIGH (operational) — Live Neon DB password present in `.env.prod`
- **Before:** Plaintext Neon password (`npg_…TQgL`) in `dashboard/.env.prod`,
  read during this audit → treat as potentially exposed.
- **After:** Rotation prepared, not executed (gated on user). Full reference
  inventory + step-by-step rotation in **`ROTATION_RUNBOOK.md`**. Confirmed the
  password lives in exactly one (git-ignored) file, is **not** in git history,
  and all other references are env-name-only. Two `VERCEL_OIDC_TOKEN` JWTs were
  also found in env files (short-lived, already expired, auto-regenerated) and
  flagged not-to-commit.

### 5. MODERATE — Transitive dependency vulns (post-bump)
- `npm audit fix` (non-breaking) cleared the **high** `picomatch` ReDoS/method-
  injection and a moderate `ws` issue.
- Bumped direct devDependency `postcss` to `^8.5.10` (clears GHSA-qx2v-qp2m-jg93
  at the project level).
- **Residual (accepted):** 2 moderate `postcss` advisories remain **only inside
  Next's own bundled copy** (`node_modules/next/node_modules/postcss`). The only
  npm-offered "fix" is `npm audit fix --force`, which **downgrades Next to 9.3.3**
  and reintroduces the 20+ High CVEs from finding #2 — **explicitly rejected**.
  This is a build-time CSS-stringify XSS, not a runtime exposure for this app;
  it resolves when Next ships an updated bundled postcss. **No action.**

---

## Residual risk / follow-ups for the user
1. **Rotate the Neon password** per `ROTATION_RUNBOOK.md` (Neon → Vercel → GitHub
   secret → local `.env.prod`).
2. **Provision `INIT_DB_TOKEN`** in Vercel (production) if runtime schema-init is
   needed; otherwise leave unset to keep the endpoint disabled.
3. Re-run `npm audit` after future Next releases to clear the bundled-postcss item.

## Verification commands used
```bash
npm install && npm run build            # → Next 16.2.9, build OK
PORT=3737 npm run start                 # then:
curl -sI http://localhost:3737/         # all security headers present
curl -s -o /dev/null -w '%{http_code}' http://localhost:3737/api/init-db        # 405 (GET)
curl -s -X POST http://localhost:3737/api/init-db                                # 503 (no token)
INIT_DB_TOKEN=… curl -s -X POST -H 'Authorization: Bearer WRONG' …/api/init-db   # 401
INIT_DB_TOKEN=… curl -s -X POST -H 'Authorization: Bearer …' …/api/init-db       # 200
```
