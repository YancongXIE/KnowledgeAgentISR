from fastapi import FastAPI

from kg_agents.web_api import app as api_app

# Keep a top-level FastAPI import and typed app binding so Azure App Service
# can auto-detect this as an ASGI/FastAPI app when startup command settings lag.
app: FastAPI = api_app
