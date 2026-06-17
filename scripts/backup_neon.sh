#!/usr/bin/env bash
#
# backup_neon.sh — versioned, compressed pg_dump of the production Neon Postgres.
#
# WHY: The prod DB has no automated dump today, and serving's sync
# (scripts/sync_sqlite_to_postgres.py) performs a destructive
# `TRUNCATE ... RESTART IDENTITY CASCADE` before reloading. A pg_dump taken
# IMMEDIATELY BEFORE any destructive prod op (and on a schedule) means a bad
# load can always be rolled back. See BACKUP_STRATEGY.md.
#
# WHAT IT DOES (read-only against the DB):
#   1. Resolves POSTGRES_URL (or --url), stripping the Vercel trailing-\n artifact.
#   2. Verifies pg_dump exists (errors with install hint if not).
#   3. Dumps to output/backups/neon/neon_<UTC-timestamp>.sql.gz (gzipped).
#   4. Validates the dump is non-empty and gunzip-decodable; deletes & fails otherwise.
#   5. Prunes dumps older than RETENTION_DAYS (default 30), but NEVER the newest.
#
# SAFETY:
#   * pg_dump is READ-ONLY — it never writes to the database.
#   * Exits non-zero on any failure so a caller (pre-op guard) can ABORT the
#     destructive op if the backup didn't succeed.
#   * Redacts credentials in all log output.
#
# USAGE:
#   POSTGRES_URL=... scripts/backup_neon.sh                 # dump now
#   scripts/backup_neon.sh --url 'postgresql://...'         # explicit URL
#   scripts/backup_neon.sh --check                          # preflight only (no dump)
#   RETENTION_DAYS=14 scripts/backup_neon.sh                # custom retention
#
# As a pre-destructive-op guard (recommended pattern for serving's sync):
#   scripts/backup_neon.sh || { echo "Backup failed — aborting load"; exit 1; }
#   python3 scripts/sync_sqlite_to_postgres.py --execute --i-have-rotated-the-secret
#
# REQUIRES: pg_dump whose MAJOR matches the Neon server (17) + gzip.
#   pg_dump CANNOT dump a server NEWER than itself — it aborts with
#   "server version mismatch". The Neon prod server is 17.x, so the client must
#   be 17.x too (NEON_SERVER_MAJOR, default 17). --check enforces this up front.
#   macOS:  brew install postgresql@17                      # provides pg_dump 17
#   Debian/Ubuntu (CI): add the PGDG repo, then
#                       sudo apt-get install -y postgresql-client-17
#   (plain `postgresql-client` on Ubuntu installs 16 — the bug this guards.)
#
set -euo pipefail

# --- locate repo paths (script lives in scrapy_project/scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKUP_DIR="${NEON_BACKUP_DIR:-${REPO_DIR}/output/backups/neon}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
# Expected Neon server major. pg_dump's major MUST equal this or it aborts with
# "server version mismatch" at dump time. Configurable so it's not a magic
# constant and tracks a future Neon upgrade. (#7 / Problem 2)
NEON_SERVER_MAJOR="${NEON_SERVER_MAJOR:-17}"

# --- args ---
CHECK_ONLY=0
URL_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --url) URL_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[backup_neon] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --- resolve POSTGRES_URL, mirroring sync_sqlite_to_postgres.py's fallbacks ---
resolve_url() {
  if [[ -n "${URL_OVERRIDE}" ]]; then printf '%s' "${URL_OVERRIDE}"; return; fi
  local v
  for v in POSTGRES_URL DATABASE_URL POSTGRES_URL_NON_POOLING; do
    if [[ -n "${!v:-}" ]]; then printf '%s' "${!v}"; return; fi
  done
  return 1
}

# Strip surrounding quotes and the Vercel-export trailing literal "\n" / real newline.
clean_url() {
  local u="$1"
  u="${u%\"}"; u="${u#\"}"           # strip wrapping double-quotes
  u="${u%\\n}"                        # strip a literal backslash-n suffix
  u="${u//$'\n'/}"; u="${u//$'\r'/}"  # strip real CR/LF
  printf '%s' "$u"
}

# pg_dump CANNOT use Neon's pgbouncer "-pooler" endpoint — transaction pooling
# drops the session state pg_dump relies on (it errors / produces nothing). Strip
# "-pooler" so the dump hits the DIRECT endpoint (Neon's documented fix). Idempotent:
# a no-op if the caller (e.g. the sync) already rewrote it, and a no-op for the
# direct/non-pooled URL form. Read-only — this only affects the dump connection.
to_direct_endpoint() {
  local u="$1"
  printf '%s' "${u/-pooler./.}"
}

# Redact user:password from a URL for safe logging -> ...@host/db
redact_url() {
  local u="$1"
  if [[ "$u" == *"@"* ]]; then printf '...@%s' "${u##*@}"; else printf '%s' "(no-host)"; fi
}

# --- preflight: tools present ---
command -v gzip >/dev/null 2>&1 || die "gzip not found."
if ! command -v pg_dump >/dev/null 2>&1; then
  die "pg_dump not found. Install the PostgreSQL ${NEON_SERVER_MAJOR} client (must match the Neon server major):
       macOS:  brew install postgresql@${NEON_SERVER_MAJOR}
       Debian: add the PGDG repo, then sudo apt-get install -y postgresql-client-${NEON_SERVER_MAJOR}"
fi

RAW_URL="$(resolve_url || true)"
[[ -n "${RAW_URL}" ]] || die "No Postgres URL. Set POSTGRES_URL (prod value lives in dashboard/.env.prod) or pass --url."
PG_URL="$(to_direct_endpoint "$(clean_url "${RAW_URL}")")"
PG_DUMP_VERSION_LINE="$(pg_dump --version 2>/dev/null | head -1)"
log "pg_dump: ${PG_DUMP_VERSION_LINE}"
log "target:  $(redact_url "${PG_URL}")"
case "${RAW_URL}" in *-pooler.*) log "note: rewrote '-pooler' → direct endpoint for pg_dump compatibility";; esac

# --- preflight: pg_dump MAJOR must match the Neon server major (#7 / Problem 2) ---
# pg_dump refuses to dump a server NEWER than itself ("server version mismatch").
# Surface that HERE — before any dump and without a live connection — so CI and a
# pre-destructive-op guard fail loudly and EARLY instead of at dump time. The real
# dump-time check below stays as the backstop. Single source of truth: scripts/
# pg_version.py (also unit-tested). Falls back to an inline parse if Python is absent.
check_pg_dump_major() {
  local helper="${SCRIPT_DIR}/pg_version.py" actual="" py=""
  for py in python3 python; do command -v "$py" >/dev/null 2>&1 && break; py=""; done
  if [[ -n "$py" && -f "$helper" ]]; then
    if "$py" "$helper" --check-major "${NEON_SERVER_MAJOR}" >/dev/null 2>&1; then
      return 0
    fi
    actual="$("$py" "$helper" --print-major 2>/dev/null || true)"
  else
    # Inline fallback: first integer after "PostgreSQL)" in the version line.
    actual="$(printf '%s' "${PG_DUMP_VERSION_LINE}" | sed -nE 's/.*PostgreSQL\) ([0-9]+).*/\1/p')"
    [[ -n "$actual" && "$actual" == "${NEON_SERVER_MAJOR}" ]] && return 0
  fi
  die "pg_dump major (${actual:-unknown}) != Neon server major (${NEON_SERVER_MAJOR}). \
pg_dump CANNOT dump a newer server — it aborts with 'server version mismatch'. \
Install the matching client:
       macOS:  brew install postgresql@${NEON_SERVER_MAJOR}
       Debian (PGDG): sudo apt-get install -y postgresql-client-${NEON_SERVER_MAJOR}
       (Override the expected major with NEON_SERVER_MAJOR if the Neon server changed.)"
}
check_pg_dump_major

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  log "preflight OK (--check): pg_dump major matches Neon server (${NEON_SERVER_MAJOR}), URL resolved. No dump taken."
  exit 0
fi

# --- take the dump ---
mkdir -p "${BACKUP_DIR}"
TS="$(date -u +%Y%m%d_%H%M%SZ)"
OUT="${BACKUP_DIR}/neon_${TS}.sql.gz"
TMP="${OUT}.partial"

log "dumping -> ${OUT}"
# --no-owner/--no-privileges keep the dump portable for restore into a fresh DB.
# pg_dump streams to stdout; gzip compresses; pipefail makes a pg_dump failure fatal.
if ! pg_dump --no-owner --no-privileges --format=plain "${PG_URL}" 2>"${BACKUP_DIR}/.last_pg_dump.err" | gzip -c > "${TMP}"; then
  rm -f "${TMP}"
  log "pg_dump stderr:"; sed 's/^/    /' "${BACKUP_DIR}/.last_pg_dump.err" >&2 || true
  die "pg_dump failed — NO backup written. (Destructive ops must NOT proceed.)"
fi

# --- validate: non-empty + gunzip-decodable + contains SQL ---
if [[ ! -s "${TMP}" ]]; then rm -f "${TMP}"; die "dump is empty — failing."; fi
if ! gzip -t "${TMP}" 2>/dev/null; then rm -f "${TMP}"; die "dump is not valid gzip — failing."; fi
# Capture the head into a var FIRST. Piping `gunzip | head -c 4096 | grep -q`
# directly inside `if !` under `set -o pipefail` is a false-failure trap: on any
# real-size dump, head closes the pipe after 4 KiB, gunzip gets SIGPIPE (141), and
# pipefail propagates that as the pipeline's status even though grep matched —
# aborting a perfectly good backup with "no recognizable SQL content". Command
# substitution + `|| true` isolates the SIGPIPE so only grep's result gates.
SQL_HEAD="$(gunzip -c "${TMP}" 2>/dev/null | head -c 4096 || true)"
if ! printf '%s' "${SQL_HEAD}" | grep -qiE 'PostgreSQL database dump|CREATE |COPY |INSERT '; then
  rm -f "${TMP}"; die "dump has no recognizable SQL content — failing."
fi

mv "${TMP}" "${OUT}"
SIZE="$(du -h "${OUT}" | cut -f1)"
log "OK: backup written (${SIZE}) -> ${OUT}"

# --- prune old dumps (keep newest regardless of age) ---
if [[ "${RETENTION_DAYS}" -gt 0 ]]; then
  # Newest dump to protect from pruning. Filenames embed a sortable UTC timestamp
  # (neon_YYYYMMDD_HHMMSSZ.sql.gz), so lexical sort == chronological sort. This is
  # portable across BSD (macOS) and GNU (CI) find without relying on -printf/stat.
  NEWEST="$(find "${BACKUP_DIR}" -name 'neon_*.sql.gz' -type f 2>/dev/null | LC_ALL=C sort | tail -1 || true)"
  pruned=0
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ "$f" == "${NEWEST}" ]] && continue   # never delete the most recent
    rm -f "$f" && pruned=$((pruned+1)) && log "pruned old: $(basename "$f")"
  done < <(find "${BACKUP_DIR}" -name 'neon_*.sql.gz' -type f -mtime "+${RETENTION_DAYS}" 2>/dev/null)
  log "retention: kept newest, pruned ${pruned} dump(s) older than ${RETENTION_DAYS}d"
fi

log "done."
