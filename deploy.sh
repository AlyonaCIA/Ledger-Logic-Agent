#!/usr/bin/env bash
# deploy.sh – build and deploy the Ledger Logic Agent to Google Cloud Run
#
# Prerequisites:
#   gcloud auth login --update-adc
#   gcloud config set project ainm26osl-714
#
# Secrets are stored in GCP Secret Manager, NOT in env files.
# Run once to create them:
#   ./deploy.sh --init-secrets
#
# Then deploy normally:
#   ./deploy.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ID="ainm26osl-714"
REGION="europe-west1"          # nearest EU region with Gemini 2.5 Flash on Vertex
SERVICE_NAME="ledger-logic-agent"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# ── Helpers ───────────────────────────────────────────────────────────────
log() { echo "▶ $*"; }

# ── Secret bootstrap (run once) ───────────────────────────────────────────
init_secrets() {
  log "Creating secrets in Secret Manager (you will be prompted for values)"

  # GEMINI_API_KEY (from https://aistudio.google.com/apikey)
  if ! gcloud secrets describe gemini-api-key --project="$PROJECT_ID" &>/dev/null; then
    printf "Paste your GEMINI_API_KEY (input hidden): "
    read -rs GEMINI_KEY_VALUE
    echo
    printf '%s' "$GEMINI_KEY_VALUE" | \
      gcloud secrets create gemini-api-key \
        --project="$PROJECT_ID" \
        --replication-policy=automatic \
        --data-file=-
    log "Secret 'gemini-api-key' created."
  else
    log "Secret 'gemini-api-key' already exists – skipping."
  fi

  # Optional: API_KEY to protect /solve
  printf "Paste your API_KEY to protect /solve (leave blank to skip): "
  read -rs CUSTOM_API_KEY
  echo
  if [[ -n "$CUSTOM_API_KEY" ]]; then
    if ! gcloud secrets describe agent-api-key --project="$PROJECT_ID" &>/dev/null; then
      printf '%s' "$CUSTOM_API_KEY" | \
        gcloud secrets create agent-api-key \
          --project="$PROJECT_ID" \
          --replication-policy=automatic \
          --data-file=-
      log "Secret 'agent-api-key' created."
    else
      # Update existing secret version
      printf '%s' "$CUSTOM_API_KEY" | \
        gcloud secrets versions add agent-api-key \
          --project="$PROJECT_ID" \
          --data-file=-
      log "Secret 'agent-api-key' updated."
    fi
  fi

  log "Done. Now run: ./deploy.sh"
}

# ── Build & push ──────────────────────────────────────────────────────────
build_and_push() {
  log "Authenticating Docker with GCR…"
  gcloud auth configure-docker --quiet

  log "Building image: ${IMAGE}…"
  docker build --platform linux/amd64 -t "${IMAGE}" .

  log "Pushing image…"
  docker push "${IMAGE}"
}

# ── Deploy to Cloud Run ───────────────────────────────────────────────────
deploy() {
  # min-instances: 1 during active competition (eliminates cold-start), else 0
  MIN_INSTANCES=${MIN_INSTANCES:-1}

  # GEMINI_API_KEY – required secret for the parser.
  # Fallback: if not present, Cloud Run uses Workload Identity (ADC) for Vertex AI.
  SECRET_FLAGS=""
  if gcloud secrets describe gemini-api-key --project="$PROJECT_ID" &>/dev/null; then
    SECRET_FLAGS="--set-secrets=GEMINI_API_KEY=gemini-api-key:latest"
  fi
  if gcloud secrets describe agent-api-key --project="$PROJECT_ID" &>/dev/null; then
    SECRET_FLAGS="${SECRET_FLAGS} --set-secrets=API_KEY=agent-api-key:latest"
  fi

  log "Deploying to Cloud Run (${REGION}) min-instances=${MIN_INSTANCES}…"
  # shellcheck disable=SC2086
  gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=1Gi \
    --cpu=1 \
    --timeout=300 \
    --concurrency=10 \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances=5 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},VERTEX_LOCATION=europe-west1" \
    ${SECRET_FLAGS}

  SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)")

  log "Deployed: ${SERVICE_URL}"
  echo "${SERVICE_URL}" > .cloud-run-url
  log "URL saved to .cloud-run-url"
}

# ── Smoke test against deployed service ──────────────────────────────────
smoke_test() {
  if [[ ! -f .cloud-run-url ]]; then
    echo "No .cloud-run-url found. Run ./deploy.sh first."
    exit 1
  fi
  URL=$(cat .cloud-run-url)
  log "Smoke-testing ${URL}…"

  # 1. Health
  STATUS=$(curl -sf "${URL}/health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))")
  echo "  /health → status=${STATUS}"
  [[ "$STATUS" == "ok" ]] || { echo "FAIL: health check"; exit 1; }

  # 2. Minimal /solve
  RESULT=$(curl -sf -X POST "${URL}/solve" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Create a customer named SmokeTest AS","files":[],"tripletex_credentials":{"base_url":"'"${TRIPLETEX_SANDBOX_URL:-https://kkpqfuj-amager.tripletex.dev/v2}"'","session_token":"'"${TRIPLETEX_SANDBOX_TOKEN:-test}"'"}}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))")
  echo "  /solve → status=${RESULT}"
  [[ "$RESULT" == "completed" ]] || { echo "FAIL: /solve did not return completed"; exit 1; }

  log "Smoke test passed ✓"
}

# ── Entry point ───────────────────────────────────────────────────────────
case "${1:-deploy}" in
  --init-secrets) init_secrets ;;
  smoke)          smoke_test ;;
  deploy)
    build_and_push
    deploy
    ;;
  *)
    echo "Usage: $0 [deploy|--init-secrets|smoke]"
    exit 1
    ;;
esac
