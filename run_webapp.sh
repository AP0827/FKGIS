#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# FKGIS — one-command start script (no database required).
#
#   ./run_webapp.sh            -> run the web app (using bundled sample outputs)
#   FKGIS_USE_LLM=true ./run_webapp.sh   -> also enable Gemini refinement
#
# The first run creates a virtualenv, installs requirements and downloads the
# spaCy transformer model used by the pipeline. Sessions are stored under
# FKGIS/webapp/sessions/ (set FKGIS_SESSIONS_DIR to override).
# -----------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

ENV_DIR=".venv"
PY=python3

if [ ! -d "$ENV_DIR" ]; then
  echo ">> Creating virtualenv at $ENV_DIR"
  "$PY" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo ">> Installing requirements"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

if ! "$PY" -m spacy validate en_core_web_trf >/dev/null 2>&1; then
  echo ">> Downloading spaCy transformer model (en_core_web_trf)"
  "$PY" -m spacy download en_core_web_trf
fi

echo ">> Starting FKGIS web app at http://${FKGIS_HOST:-127.0.0.1}:${FKGIS_PORT:-8000}"
exec "$PY" -m uvicorn FKGIS.webapp.main:app \
  --host "${FKGIS_HOST:-127.0.0.1}" \
  --port "${FKGIS_PORT:-8000}" \
  "$@"