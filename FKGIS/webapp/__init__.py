"""FKGIS web application package."""

# Intentionally does not import ``main`` here so that importing submodules
# (e.g. ``FKGIS.webapp.case_manager``) does not trigger FastAPI app creation.
# Use ``from FKGIS.webapp.main import app`` when you need the application object.