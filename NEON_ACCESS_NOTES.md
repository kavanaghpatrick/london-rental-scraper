# NEON RECOVERY — ACCESS & REACHABILITY NOTES (security assist for #31)

**Purpose:** Answer the one question the recovery hinges on operationally —
*what can we do programmatically with the credentials we already have, vs. what
strictly requires the USER's browser (Neon/Vercel console)?* Companion to
serving's `NEON_RECOVERY_RUNBOOK.md` (which owns the recovery procedure).

**Owner:** security · **Date:** 2026-06-16

---

## TL;DR — the recovery is a CONSOLE (browser) action, not a scriptable one

We hold a **database credential** (`POSTGRES_URL`), not a **control-plane
credential**. PITR / branch-from-timestamp / restore are control-plane
operations. Therefore **the user must do the recovery in the browser console**
(or first mint a `NEON_API_KEY`). Nothing security or serving can run with the
current secrets will create the recovery branch.

---

## What credentials actually exist (audited)

| Credential | Present? | What it can do | What it CANNOT do |
|---|---|---|---|
| `POSTGRES_URL` (in `dashboard/.env.prod`) | ✅ yes | Connect to the DB, run SQL on the **current** (post-truncate) data | ❌ branch, restore, PITR, read history, see plan/retention — these are control-plane, not SQL |
| `NEON_API_KEY` (`napi_…`) | ❌ **NOT in repo** (grepped `.env*`, `*.sh`, `*.yml`, `*.py`, `*.ts`) | (would enable neonctl / Management API branching) | n/a — does not exist yet |
| `VERCEL_OIDC_TOKEN` | ✅ yes (expired, ~12h JWT) | Vercel OIDC identity for the `dashboard` project | ❌ not a Neon credential; cannot manage Neon |
| `neonctl` CLI | ❌ not installed | — | — |

**Plan tier (confirms the deadline):** the Vercel OIDC token's `plan` claim is
**`hobby`**. The DB is provisioned via the **Vercel Postgres → Neon integration**
(`@vercel/postgres` in `dashboard/package.json`). A Hobby/Vercel-Postgres Neon
project maps to Neon's **Free tier → ~6-hour PITR history window.** If the
truncate was >~6h ago, PITR is likely already expired (serving's runbook §1/§5).

---

## Reachable PROGRAMMATICALLY (what we/serving can do now, read-only)

With `POSTGRES_URL` alone, over the Postgres protocol:

- ✅ Confirm the **current** clobbered state: `SELECT COUNT(*) FROM listings;`
  (expect ~10,048 — the post-truncate value).
- ✅ Read schema, current `scrape_runs`, max `last_seen`, etc.
- ✅ Take a `pg_dump` of the **current** state (this is what `scripts/backup_neon.sh`
  does — but it can only capture post-truncate data; it cannot reach history).
- ❌ **Cannot** time-travel-query a pre-truncate timestamp. Neon's time travel is a
  **console/SQL-Editor feature bound to a branch + history**, not something the raw
  connection string exposes to an external psql client.
- ❌ **Cannot** create a branch, restore, or read the history window.

> Implication: there is **no script** security can hand serving that recovers the
> data. The recovery branch must be created in the console or via neonctl+API key.

## Requires the USER's CONSOLE / browser (control-plane)

All of these need either (a) browser login to the console, or (b) a `NEON_API_KEY`
the user mints:

1. **Checking plan + history window** (the deadline) — console only.
2. **Time Travel verify** the pre-truncate count (~22,317) — console SQL Editor.
3. **Create branch from timestamp** before the truncate — console or neonctl+API key.
4. **Restore prod from history** — console or API key.

### Two consoles — try Vercel first (likely fastest for this setup)
Because the DB is **Vercel-Postgres-provisioned**, the user has two entry points:

- **A. Vercel Dashboard → Storage** (vercel.com → project `dashboard` →
  **Storage** tab → the Postgres store). Vercel surfaces a **Neon-backed restore /
  branching UI** here directly, and it's the account the user already logs into.
  **Start here.**
- **B. Neon Console** (https://console.neon.tech) — the same project may also be
  reachable directly. From Vercel Storage there is usually an **"Open in Neon"**
  link. Use this if the Vercel UI doesn't expose branching.

Project markers to confirm you're in the right place: endpoint
`ep-fancy-union-adjjk0kb`, database `neondb`, region `us-east-1`.

---

## If the user wants the CLI/API path (to make it scriptable)

This is the ONLY way to make recovery programmatic — it requires a one-time
human step to mint the key:

1. **User mints a Neon API key:** Neon Console → **Account settings → API keys →
   Generate** (or via the Vercel→Neon link). Value looks like `napi_…`.
2. Provide it to the operator as `NEON_API_KEY` (env var — do NOT commit; it goes
   in the same git-ignored `.env.prod` / a secret store, same handling as
   `POSTGRES_URL`).
3. Then neonctl/Management API can branch + restore (commands in serving's runbook §2).

> security note: a `NEON_API_KEY` is **more powerful than `POSTGRES_URL`** (it can
> delete the whole project). Treat it as top-secret, scope/rotate it after the
> recovery, and prefer the console for a one-shot recovery rather than minting a
> long-lived key.

---

## What security IS doing here (vs. what's blocked on the user)

| Action | Who | State |
|---|---|---|
| Audit which creds exist / are reachable | security | ✅ done (this doc) |
| Confirm plan tier / deadline driver | security | ✅ `hobby` → Free → ~6h window |
| Create the recovery branch | **USER** (console) | ⛔ blocked — needs browser, not scriptable with current creds |
| Validate recovered branch data | dataeng | pending the branch |
| Take a current-state `pg_dump` safety net | security | ⛔ gated by lead's "no live-Neon ops until rotation" order — script ready (`backup_neon.sh`) |
| Rotate `POSTGRES_URL` after recovery | **USER** | pending (`dashboard/ROTATION_RUNBOOK.md`) |

**Bottom line for the lead:** there is nothing to "try programmatically" first —
hand the user serving's runbook §1–§2 (or the Vercel-Storage path above) and have
them do it in the browser **now**, because the 6-hour Free-tier window is the
binding constraint. If the window has already closed, jump to runbook §5 fallbacks
(git-history reconstruction — already largely done in #34/#35).

---

## OUTCOME + STANDING LESSON (post-mortem, 2026-06-16)

**Recovery FAILED — the window had closed.** The user checked the console: Free tier,
6-hour window, truncate ~18h prior, no snapshots → no pre-truncate restore point.
~12,269 listings + ~18,980 price-history rows are permanently lost; rebuild is via a
fresh scrape (#41) + merge (#42).

**The access gap, stated plainly (this is the lesson):** when it mattered, we held
only a **data-plane** credential (`POSTGRES_URL`) and **no control-plane** credential
(`NEON_API_KEY`). So we could not branch / PITR / restore programmatically, and the
recovery was 100% gated on a human in a browser — which lost time against a 6h clock.

**Two forward-looking fixes (recommended; tie into #38 backup regime):**
1. **User mints a `NEON_API_KEY`** (Neon Console → Account → API keys), stored like
   `POSTGRES_URL` (git-ignored / secret store). Then future recovery IS scriptable
   via `neonctl branches create --timestamp ...`. Caveat from §"CLI path": an API key
   can delete the whole project — scope it, and rotate after use.
2. **Upgrade off Free tier** — a 6h PITR window is brutal for a prod DB. Neon Launch
   (7-day window) is the prominent TOP recommendation in `BACKUP_STRATEGY.md`. With it,
   an accidental write is recoverable for a week instead of 6 hours.

Until both land, **`scripts/backup_neon.sh` (L3) + the nightly `neon-backup.yml` (L2)
are the real safety net** — and the #37 sync now runs the L3 dump fail-closed before
every write. So we are materially safer going forward even staying on Free tier.
