#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# FKGIS — start script (no database required, no installs).
#
#   ./run_webapp.sh            -> run the web app (using bundled sample outputs)
#   FKGIS_USE_LLM=true ./run_webapp.sh   -> also enable Gemini refinement
#
# Assumes dependencies are already installed (pip install -r requirements.txt,
# plus `python -m spacy download en_core_web_trf` to Run Pipeline). Uses the
# project virtualenv (./.venv) when present, otherwise the system python3.
# Sessions are stored under FKGIS/webapp/sessions/ (set FKGIS_SESSIONS_DIR to
# override).
# -----------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo ">> Starting FKGIS web app at http://${FKGIS_HOST:-127.0.0.1}:${FKGIS_PORT:-8000}"
exec "$PY" -m uvicorn FKGIS.webapp.main:app \
  --host "${FKGIS_HOST:-127.0.0.1}" \
  --port "${FKGIS_PORT:-8000}" \
  "$@"