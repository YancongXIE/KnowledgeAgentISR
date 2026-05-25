# T-Rex ISR Agent Backend

This directory contains the Python backend for the T-Rex ISR agent.

## Local run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API locally:

```bash
python3 -m uvicorn kg_agents.web_api:app --reload
```

## Azure Web App

This backend is prepared for Azure App Service (Linux).

- App entrypoint: `app:app`
- Startup script: `startup.sh`

Recommended startup command in Azure Web App configuration:

```bash
bash startup.sh
```

Required app settings include:

```bash
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
AZURE_ENDPOINT=...
AZURE_API_KEY=...
AZURE_API_VERSION=2024-12-01-preview
AZURE_MODEL_NAME=gpt-5.2
AZURE_MODEL_DEPLOYMENT=gpt-5.2
WEB_ALLOWED_ORIGINS=...
```
