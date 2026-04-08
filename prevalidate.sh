#!/usr/bin/env bash

set -e

REPO_DIR=$(pwd)
DOCKER_BUILD_TIMEOUT=300

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

pass() {
  echo -e "${GREEN}✔ $1${NC}"
}

fail() {
  echo -e "${RED}✖ $1${NC}"
}

log() {
  echo -e "$1"
}

stop_at() {
  echo -e "\n${RED}${BOLD}Stopped at $1${NC}"
  exit 1
}

run_with_timeout() {
  timeout "$1" "${@:2}"
}

echo ""
echo "========================================"
echo "  OpenEnv Pre-validation Script"
echo "========================================"
echo ""

# -------------------------
# STEP 1: Check files
# -------------------------

log "${BOLD}Step 1/3: Checking required files...${NC}"

REQUIRED_FILES=("inference.py" "openenv.yaml" "Dockerfile" "requirements.txt")

for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/$file" ]; then
    fail "$file not found in root directory"
    stop_at "Step 1"
  fi
done

pass "All required files present"

# -------------------------
# STEP 2: Docker build
# -------------------------

log "${BOLD}Step 2/3: Running docker build...${NC}"

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  echo "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

BUILD_OK=false

if BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t openenv-test . 2>&1); then
  BUILD_OK=true
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  echo "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

# -------------------------
# STEP 3: OpenEnv validate
# -------------------------

log "${BOLD}Step 3/3: Running openenv validate...${NC}"

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  echo "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false

if VALIDATE_OUTPUT=$(openenv validate 2>&1); then
  VALIDATE_OK=true
fi

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && echo "$VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  echo "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

echo ""
echo "========================================"
echo -e "${GREEN}${BOLD} All checks passed! Ready to submit.${NC}"
echo "========================================"
echo ""

exit 0