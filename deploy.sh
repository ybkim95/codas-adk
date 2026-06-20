#!/bin/bash
# CoDaS - Cloud Run Deployment Script
# This script builds and deploys the CoDaS service to Google Cloud Run with sane defaults.

set -euo pipefail

# === CONFIGURATION BLOCK ===
# Edit these values, or override any of them from the environment
# (e.g. PROJECT_ID=my-project REGION=us-east1 ./deploy.sh).
# Leave PROJECT_ID empty to use the active gcloud project.
PROJECT_ID="${PROJECT_ID:-}"
SERVICE_NAME="${SERVICE_NAME:-codas-service}"
REGION="${REGION:-us-central1}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"

# CoDaS API Authentication Key (For securing access to the Cloud Run endpoint).
# If left empty, a secure key will be automatically generated during deployment.
CODAS_AGENT_API_KEYS="${CODAS_AGENT_API_KEYS:-}"

# Gemini API Authentication Options:
# Option A: Use a Google AI Studio API Key. Paste it here.
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

# Option B: Use Vertex AI native integration (Recommended).
# Set to "TRUE" to use a dedicated GCP Service Account with Vertex AI access.
GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-TRUE}"

# Session Backend configuration ('memory' or 'vertex')
# 'memory' keeps sessions local to the container (requires MAX_INSTANCES=1).
# 'vertex' uses Vertex AI Agent Engine to support scaling (requires Vertex AI API enabled).
CODAS_SESSION_BACKEND="${CODAS_SESSION_BACKEND:-memory}"

# IAM Authentication (highly recommended)
# Set to "no" to disable public internet access and restrict callers to authorized IAM identities.
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-no}"

# Custom Service Account for Cloud Run. If empty, the script will automatically
# create and configure a dedicated 'codas-runner' service account for you.
CODAS_RUN_SERVICE_ACCOUNT="${CODAS_RUN_SERVICE_ACCOUNT:-}"
# ============================

# ANSI Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CoDaS Cloud Run Deployment CLI ===${NC}"

# Check for gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: 'gcloud' CLI is not installed.${NC}"
    echo "Please install the Google Cloud SDK first: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Apply Project Configuration
if [ -n "$PROJECT_ID" ]; then
    echo -e "Setting target GCP Project to: ${GREEN}${PROJECT_ID}${NC}"
    gcloud config set project "$PROJECT_ID" &>/dev/null || true
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No active project configured in gcloud.${NC}"
    exit 1
fi

echo -e "Deploying in GCP Project: ${GREEN}${PROJECT_ID}${NC}"

# Configure Service Authorization Key
if [ -z "${CODAS_AGENT_API_KEYS:-}" ]; then
    echo -e "${YELLOW}CODAS_AGENT_API_KEYS is not set.${NC}"
    # Generate a secure random key
    GENERATED_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo -e "Generating a secure API key for service authentication: ${GREEN}${GENERATED_KEY}${NC}"
    echo -e "${YELLOW}IMPORTANT: Save this key! You must pass this key in the 'x-codas-agent-key' header (or as Bearer token) to access the API.${NC}"
    CODAS_AGENT_API_KEYS="$GENERATED_KEY"
else
    echo -e "Using configured CODAS_AGENT_API_KEYS from environment."
fi

# Configure Gemini Connection
GEMINI_AUTH_VARS=""
GEMINI_USES_VERTEX="no"
if [ -n "${GOOGLE_API_KEY:-}" ]; then
    echo -e "Configuring Gemini client via ${GREEN}GOOGLE_API_KEY${NC}."
    GEMINI_AUTH_VARS="GOOGLE_API_KEY=${GOOGLE_API_KEY}"
elif [ "${GOOGLE_GENAI_USE_VERTEXAI:-}" = "TRUE" ] || [ "${GOOGLE_GENAI_USE_VERTEXAI:-}" = "true" ]; then
    echo -e "Configuring Gemini client via ${GREEN}Vertex AI (Application Default Credentials)${NC}."
    GEMINI_AUTH_VARS="GOOGLE_GENAI_USE_VERTEXAI=TRUE"
    GEMINI_USES_VERTEX="yes"
else
    echo -e "${YELLOW}WARNING: No GOOGLE_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=TRUE detected in your environment.${NC}"
    echo "The service will deploy, but the agent endpoint (/v1/agent) will return 503 (unconfigured)."
    echo "To configure, export GOOGLE_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=TRUE before running this script."
fi

# Enable necessary APIs
echo -e "${BLUE}Enabling Google Cloud APIs (Cloud Run, Cloud Build)...${NC}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID"

# Build deployment environment variables list
ENV_VARS="CODAS_AGENT_API_KEYS=${CODAS_AGENT_API_KEYS}"
if [ -n "$GEMINI_AUTH_VARS" ]; then
    ENV_VARS="${ENV_VARS},${GEMINI_AUTH_VARS}"
fi

# Service Account and IAM setup (only when something actually talks to Vertex AI)
SA_EMAIL=""
if [ "$GEMINI_USES_VERTEX" = "yes" ] || [ "${CODAS_SESSION_BACKEND:-}" = "vertex" ]; then
    if [ -n "${CODAS_RUN_SERVICE_ACCOUNT:-}" ]; then
        SA_EMAIL="$CODAS_RUN_SERVICE_ACCOUNT"
        echo -e "Using user-specified service account: ${GREEN}${SA_EMAIL}${NC}"
    else
        SA_NAME="codas-runner"
        SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
        echo -e "Configuring dedicated service account: ${GREEN}${SA_EMAIL}${NC}"
        
        # Enable IAM API
        gcloud services enable iam.googleapis.com --project="$PROJECT_ID"
        
        # Create service account if it doesn't exist
        if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
            echo -e "Creating service account '${SA_NAME}'..."
            gcloud iam service-accounts create "$SA_NAME" \
                --description="Service account for CoDaS Cloud Run service" \
                --display-name="CoDaS Runner" \
                --project="$PROJECT_ID"
        fi
        
        # Grant Vertex AI User role
        echo -e "Granting ${BLUE}roles/aiplatform.user${NC} to service account..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="roles/aiplatform.user" >/dev/null || echo -e "${YELLOW}Warning: Failed to grant 'roles/aiplatform.user' to ${SA_EMAIL}. If the deployment fails to call Gemini, check your project IAM permissions.${NC}"
    fi
    SA_FLAG="--service-account=${SA_EMAIL}"
else
    SA_FLAG=""
fi

# Vertex AI (used by Gemini and/or the session service) needs its API enabled and the
# project/location passed to the container. Without this the service still deploys, but the
# first Gemini call fails on a project that has not already enabled aiplatform.googleapis.com.
if [ "$GEMINI_USES_VERTEX" = "yes" ] || [ "${CODAS_SESSION_BACKEND:-}" = "vertex" ]; then
    echo -e "${BLUE}Enabling Vertex AI API...${NC}"
    gcloud services enable aiplatform.googleapis.com --project="$PROJECT_ID"
    ENV_VARS="${ENV_VARS},GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}"
fi

# Session backend
if [ "${CODAS_SESSION_BACKEND:-}" = "vertex" ]; then
    echo -e "Configuring session backend to ${GREEN}Vertex AI Agent Engine Session Service${NC}."
    ENV_VARS="${ENV_VARS},CODAS_SESSION_BACKEND=vertex"
else
    echo -e "Configuring session backend to ${GREEN}InMemory (Single Instance)${NC}."
    echo -e "Setting max instances to ${GREEN}1${NC} to prevent state divergence."
    MAX_INSTANCES=1
fi

# Perform Deployment
echo -e "${BLUE}Starting Cloud Run deployment of '${SERVICE_NAME}' in region '${REGION}'...${NC}"

# Determine authentication flag
if [ "$ALLOW_UNAUTHENTICATED" = "yes" ] || [ "$ALLOW_UNAUTHENTICATED" = "true" ]; then
    AUTH_FLAG="--allow-unauthenticated"
else
    AUTH_FLAG="--no-allow-unauthenticated"
fi

# Run deploy
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --max-instances="$MAX_INSTANCES" \
    $AUTH_FLAG \
    --set-env-vars="$ENV_VARS" \
    $SA_FLAG \
    --project="$PROJECT_ID"

echo -e "\n${GREEN}✔ Service deployed successfully!${NC}"
echo -e "To access the service, use the URL printed above."
echo -e "Include the header: ${BLUE}x-codas-agent-key: ${CODAS_AGENT_API_KEYS}${NC}"

if [ "$ALLOW_UNAUTHENTICATED" != "yes" ] && [ "$ALLOW_UNAUTHENTICATED" != "true" ]; then
    echo -e "\n${YELLOW}🔒 Service is secured via IAM.${NC}"
    echo "To test it locally, include your gcloud identity token in the Authorization header:"
    echo -e "e.g., ${BLUE}curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" -H \"x-codas-agent-key: ${CODAS_AGENT_API_KEYS}\" [URL]/health${NC}"
fi
