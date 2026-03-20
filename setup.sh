#!/usr/bin/env bash
# Octane Setup — Apple Silicon (M-series)
# Fresh-clone to green in one run.
# Usage: bash setup.sh [--skip-brew] [--skip-playwright]

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC}  $*"; }
warn() { echo -e "  ${YELLOW}!${NC}  $*"; }
info() { echo -e "  ${BLUE}→${NC}  $*"; }
fail() { echo -e "  ${RED}✗${NC}  $*"; exit 1; }

echo ""
echo -e "${BOLD}${BLUE}🔥  Octane Setup${NC}  —  Apple Silicon"
echo -e "    $(date '+%Y-%m-%d %H:%M')"
echo ""

# ── Parse flags ────────────────────────────────────────────────────────────────
SKIP_BREW=0
SKIP_PLAYWRIGHT=0
for arg in "$@"; do
  case $arg in
    --skip-brew)       SKIP_BREW=1 ;;
    --skip-playwright) SKIP_PLAYWRIGHT=1 ;;
  esac
done

# ── 0. Resolve helpers ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

# Homebrew prefix (works on both Apple Silicon and Intel)
BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

PG_VERSION="16"
PG_BIN="${BREW_PREFIX}/opt/postgresql@${PG_VERSION}/bin"

# ── 1. Homebrew ─────────────────────────────────────────────────────────────────
if [[ $SKIP_BREW -eq 0 ]]; then
  echo -e "${BOLD}── Services (Homebrew) ──────────────────────────────────────${NC}"

  command -v brew &>/dev/null || fail "Homebrew not found. Install: https://brew.sh"
  ok "Homebrew $(brew --version | head -1 | awk '{print $2}')"

  # ── Redis ────────────────────────────────────────────────────────────────────
  if brew list redis &>/dev/null 2>&1; then
    ok "Redis already installed"
  else
    info "Installing Redis..."
    brew install redis
    ok "Redis installed"
  fi

  info "Starting Redis service..."
  brew services start redis 2>/dev/null || brew services restart redis 2>/dev/null || true
  sleep 1
  if redis-cli ping 2>/dev/null | grep -q PONG; then
    ok "Redis responding (PONG)"
  else
    warn "Redis not responding — may need a moment. Try: redis-cli ping"
  fi

  # ── PostgreSQL ───────────────────────────────────────────────────────────────
  if brew list "postgresql@${PG_VERSION}" &>/dev/null 2>&1; then
    ok "PostgreSQL ${PG_VERSION} already installed"
  else
    info "Installing PostgreSQL ${PG_VERSION}..."
    brew install "postgresql@${PG_VERSION}"
    ok "PostgreSQL ${PG_VERSION} installed"
  fi

  # Add postgres bin to PATH for this script
  export PATH="${PG_BIN}:${PATH}"

  info "Starting PostgreSQL service..."
  brew services start "postgresql@${PG_VERSION}" 2>/dev/null \
    || brew services restart "postgresql@${PG_VERSION}" 2>/dev/null || true
  sleep 2

  if "${PG_BIN}/pg_isready" -q 2>/dev/null; then
    ok "PostgreSQL accepting connections"
  else
    warn "PostgreSQL not ready yet — waiting 3 more seconds..."
    sleep 3
    "${PG_BIN}/pg_isready" -q 2>/dev/null || warn "PostgreSQL still not ready. Check: brew services list"
  fi

  # ── Create database ───────────────────────────────────────────────────────────
  DB_EXISTS="$("${PG_BIN}/psql" -U "$(whoami)" -tAc "SELECT 1 FROM pg_database WHERE datname='octane'" postgres 2>/dev/null || echo "")"
  if [[ "${DB_EXISTS}" == "1" ]]; then
    ok "Database 'octane' already exists"
  else
    info "Creating database 'octane'..."
    "${PG_BIN}/createdb" octane 2>/dev/null \
      || "${PG_BIN}/psql" -U "$(whoami)" -c "CREATE DATABASE octane;" postgres 2>/dev/null \
      || warn "Could not create database 'octane' — you may need to run: createdb octane"
    ok "Database 'octane' created"
  fi

  echo ""
fi  # end SKIP_BREW

# ── 2. Python environment ──────────────────────────────────────────────────────
echo -e "${BOLD}── Python ──────────────────────────────────────────────────${NC}"

# Ensure venv exists
if [[ ! -f "${REPO_ROOT}/.venv/bin/python3" ]]; then
  info "Creating virtual environment..."
  cd "${REPO_ROOT}"
  python3 -m venv .venv
  ok "Virtual environment created"
else
  ok "Virtual environment exists"
fi

# Bootstrap pip inside venv
info "Bootstrapping pip..."
"${PYTHON}" -m ensurepip --upgrade -q 2>/dev/null || true
"${PYTHON}" -m pip install --upgrade pip -q

# Install the package in editable mode
info "Installing octane + dependencies..."
cd "${REPO_ROOT}"
"${PYTHON}" -m pip install -e "." -q

# Install additional deps not yet in pyproject.toml
MISSING_DEPS=""
"${PYTHON}" -c "import matplotlib" 2>/dev/null || MISSING_DEPS="${MISSING_DEPS} matplotlib>=3.7"
"${PYTHON}" -c "import numpy"      2>/dev/null || MISSING_DEPS="${MISSING_DEPS} numpy>=1.24"

if [[ -n "${MISSING_DEPS}" ]]; then
  info "Installing optional deps:${MISSING_DEPS}"
  # shellcheck disable=SC2086
  "${PYTHON}" -m pip install ${MISSING_DEPS} -q
  ok "Optional deps installed"
fi

# Dev deps (pytest, etc.)
info "Installing dev dependencies..."
"${PYTHON}" -m pip install \
  pytest>=8.0 \
  pytest-asyncio>=0.24 \
  ruff \
  -q
ok "Dev dependencies installed"

echo ""

# ── 3. Playwright browsers ─────────────────────────────────────────────────────
if [[ $SKIP_PLAYWRIGHT -eq 0 ]]; then
  echo -e "${BOLD}── Playwright ──────────────────────────────────────────────${NC}"
  
  # Check if chromium is already installed for playwright
  if "${PYTHON}" -m playwright install --dry-run chromium 2>&1 | grep -q "already installed" 2>/dev/null \
     || "${PYTHON}" -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium; print('ok' if b else '')" 2>/dev/null | grep -q ok; then
    ok "Playwright Chromium already installed"
  else
    info "Installing Playwright Chromium (headless browser)..."
    "${PYTHON}" -m playwright install chromium 2>/dev/null \
      || warn "Playwright browser install failed — browser tests will be skipped"
    ok "Playwright Chromium installed"
  fi
  echo ""
fi

# ── 4. .env ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}── Configuration ───────────────────────────────────────────${NC}"

ENV_FILE="${REPO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  info "Creating .env from defaults..."
  cat > "${ENV_FILE}" << 'ENVEOF'
# Octane Environment Configuration
# Generated by setup.sh — edit as needed

# ── Bodega Inference Engine ─────────────────────────────────────
BODEGA_INFERENCE_URL=http://localhost:44468
BODEGA_TOPOLOGY=auto

# ── Bodega Intelligence (search, finance, news) ──────────────────
BODEGA_INTEL_URL=http://localhost:44469
BODEGA_INTEL_API_KEY=

# ── Storage ──────────────────────────────────────────────────────
POSTGRES_URL=postgresql://localhost:5432/octane
REDIS_URL=redis://localhost:6379/0

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FORMAT=console
ENVEOF
  ok ".env created"
else
  ok ".env exists (not overwriting)"
fi

# ── 5. Smoke test ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Smoke Test ──────────────────────────────────────────────${NC}"

# Redis
if redis-cli ping 2>/dev/null | grep -q PONG; then
  ok "Redis:     localhost:6379 ✓"
else
  warn "Redis:     not responding (start: brew services start redis)"
fi

# Postgres
if "${PG_BIN}/pg_isready" -q 2>/dev/null; then
  ok "Postgres:  localhost:5432 ✓"
else
  warn "Postgres:  not responding (start: brew services start postgresql@${PG_VERSION})"
fi

# Bodega
BODEGA_STATUS=$(curl -sf http://localhost:44468/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "unreachable")
if [[ "${BODEGA_STATUS}" == "ok" ]]; then
  MODEL_COUNT=$(curl -sf http://localhost:44468/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models_detail',[])))" 2>/dev/null || echo "?")
  ok "Bodega:    localhost:44468 ✓  (${MODEL_COUNT} model(s) loaded)"
else
  warn "Bodega:    not running — start it separately, then run: octane health"
fi

# ── 6. Final summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}✅  Setup complete${NC}"
echo ""
echo -e "  Commands available via: ${YELLOW}.venv/bin/octane${NC}  or  ${YELLOW}python3 -m octane${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo -e "    ${YELLOW}python3 -m octane health${NC}                       # system status"
echo -e "    ${YELLOW}python3 -m octane ask 'NVDA stock price'${NC}       # single query"
echo -e "    ${YELLOW}python3 -m octane investigate 'Is NVDA overvalued'${NC} # deep research"
echo ""
echo -e "  ${BOLD}Run tests:${NC}"
echo -e "    ${YELLOW}.venv/bin/python3 -m pytest -q${NC}"
echo ""
