#!/usr/bin/env bash
# =============================================================================
# Aegis-1 — Huawei ECS Deployment Script
# Ubuntu 22.04 / 24.04
# =============================================================================
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Required env vars (set before running):
#   export DEEPSEEK_API_KEY=sk-...
#   export OBS_ACCESS_KEY=...       (optional — Huawei OBS)
#   export OBS_SECRET_KEY=...       (optional)
#   export OBS_BUCKET=...           (optional)
#   export OBS_ENDPOINT=obs.af-south-1.myhuaweicloud.com  (optional)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
fail() { echo -e "${RED}❌ $*${NC}"; exit 1; }

echo -e "\n${BOLD}🧠  Aegis-1 — Huawei ECS Deployment${NC}\n"

# ── 1. Check API key ──────────────────────────────────────────────────────────
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && fail "DEEPSEEK_API_KEY not set. Run: export DEEPSEEK_API_KEY=sk-..."
ok "API key found"

# ── 2. System dependencies ────────────────────────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git curl wget build-essential \
    libssl-dev libffi-dev python3-dev
ok "System dependencies installed"

# ── 3. Python virtual environment ─────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

log "Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
ok "Virtual environment ready"

# ── 4. Python dependencies ────────────────────────────────────────────────────
log "Installing Python dependencies..."
pip install -q \
    openai \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    sentence-transformers \
    requests \
    nengo \
    torch \
    esdk-obs-python \
    pydantic
ok "Python dependencies installed"

# ── 5. Verify src/ directory ──────────────────────────────────────────────────
[[ ! -d "$PROJECT_DIR/src" ]] && fail "src/ directory not found. Run the bundle extraction script first."
ok "src/ directory found"

# ── 6. Create systemd service ─────────────────────────────────────────────────
log "Creating systemd service..."

SERVICE_FILE="$HOME/.config/systemd/user/aegis.service"
mkdir -p "$HOME/.config/systemd/user"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Aegis-1 Cognitive AI API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
Environment=DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
Environment=OBS_ACCESS_KEY=${OBS_ACCESS_KEY:-}
Environment=OBS_SECRET_KEY=${OBS_SECRET_KEY:-}
Environment=OBS_BUCKET=${OBS_BUCKET:-}
Environment=OBS_ENDPOINT=${OBS_ENDPOINT:-}
ExecStart=$VENV_DIR/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aegis

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now aegis
ok "Systemd service created and started"

# ── 7. Configure firewall ─────────────────────────────────────────────────────
log "Configuring firewall..."
if command -v ufw &>/dev/null; then
    sudo ufw allow 8000/tcp comment "Aegis-1 API" 2>/dev/null || warn "ufw rule skipped"
fi
# Note: also open port 8000 in Huawei Cloud ECS Security Group rules
warn "Remember to open port 8000 in your Huawei ECS Security Group"
ok "Firewall configured"

# ── 8. Health check ───────────────────────────────────────────────────────────
log "Waiting for Aegis-1 to boot (30s)..."
sleep 30

MAX_TRIES=6
for i in $(seq 1 $MAX_TRIES); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        ok "Aegis-1 is running and healthy!"
        break
    fi
    [[ $i -eq $MAX_TRIES ]] && warn "Health check timed out — check: journalctl --user -u aegis -f"
    sleep 10
done

# ── 9. Done ───────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_ECS_IP")

echo ""
echo -e "${BOLD}${GREEN}🎉  Deployment complete!${NC}"
echo ""
echo -e "  Web UI    : ${BOLD}http://$PUBLIC_IP:8000${NC}"
echo -e "  API docs  : ${BOLD}http://$PUBLIC_IP:8000/docs${NC}"
echo -e "  Health    : ${BOLD}http://$PUBLIC_IP:8000/health${NC}"
echo ""
echo -e "  Manage service:"
echo -e "    systemctl --user status aegis"
echo -e "    systemctl --user restart aegis"
echo -e "    journalctl --user -u aegis -f   ${YELLOW}(live logs)${NC}"
echo ""