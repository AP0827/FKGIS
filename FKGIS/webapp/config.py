"""Application configuration for the FKGIS web app.

All settings are derived from environment variables (or a local ``.env``) so
the app runs with zero database setup. Secrets such as ``GEMINI_API_KEY`` are
read from the environment and never hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# The directory that contains the ``webapp`` package (i.e. the FKGIS package root)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

try:  # optional convenience: load GEMINI_API_KEY / FKGIS_* from a local .env
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


@dataclass
class Settings:
    # Session / case workspaces live here. Each case gets its own subdirectory.
    sessions_dir: Path = field(
        default_factory=lambda: Path(os.getenv("FKGIS_SESSIONS_DIR", str(PACKAGE_ROOT / "webapp" / "sessions")))
    )
    # Bundled case documents shipped with the NLP pipeline.
    bundled_case_docs: Path = field(
        default_factory=lambda: Path(os.getenv("FKGIS_BUNDLED_DOCS", str(PACKAGE_ROOT / "nlp_pipeline" / "case_docs")))
    )
    # Bundled outputs from a prior run, used to seed the sample case instantly.
    bundled_outputs: Path = field(
        default_factory=lambda: Path(os.getenv("FKGIS_BUNDLED_OUTPUTS", str(PACKAGE_ROOT / "nlp_pipeline" / "output")))
    )

    # LLM refinement (Gemini, same model family as the current implementation).
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    use_llm: bool = field(default_factory=lambda: os.getenv("FKGIS_USE_LLM", "false").lower() in {"1", "true", "yes"})

    host: str = field(default_factory=lambda: os.getenv("FKGIS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("FKGIS_PORT", "8000")))

    def ensure_dirs(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings