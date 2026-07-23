"""
Runtime harness for KDE live quality tests.

Probes Azure / Neo4j availability and builds agent instances without
requiring Neo4j for non-retrieval agents.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import MagicMock

from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg_agents.elicitation_agent import ElicitationAgent
from kg_agents.extraction_agent import ExtractionAgent
from kg_agents.integration_agent import IntegrationAgent
from kg_agents.orchestrator import AzureOpenAIClient, KGMultiAgentOrchestrator
from kg_agents.reflection_agent import ReflectionAgent
from kg_agents.vicarious_learning_agent import VicariousLearningAgent


def load_env() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (".env", "kg_agents/.env"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            load_dotenv(path)
            break
    else:
        load_dotenv()
    # Live quality tests need larger JSON payloads for IntegrationState / VicariousLearningOutput.
    os.environ.setdefault("KG_JSON_MAX_TOKENS", "4096")


def azure_configured() -> bool:
    load_env()
    return bool(os.environ.get("AZURE_ENDPOINT") and os.environ.get("AZURE_API_KEY"))


def build_azure_client() -> AzureOpenAIClient:
    load_env()
    endpoint = os.environ["AZURE_ENDPOINT"].strip()
    api_key = os.environ["AZURE_API_KEY"].strip()
    api_version = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
    deployment = os.environ.get("AZURE_MODEL_DEPLOYMENT", "gpt-5.2")
    return AzureOpenAIClient(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment=deployment,
    )


def probe_neo4j(timeout_s: float = 5.0) -> Tuple[bool, str]:
    load_env()
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not uri or not user or not password:
        return False, "NEO4J_* env vars missing"

    try:
        from urllib.parse import urlparse

        parsed = urlparse(uri.replace("neo4j+s://", "https://").replace("neo4j://", "http://"))
        host = parsed.hostname
        if not host:
            return False, f"invalid NEO4J_URI: {uri}"
        socket.getaddrinfo(host, 7687)
    except OSError as exc:
        return False, f"DNS/resolve failed: {exc}"

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return True, f"connected to {uri}"
    except Exception as exc:
        return False, f"connect failed: {type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            driver.close()


def build_embed_fn() -> Callable[[List[str]], List[List[float]]]:
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    return embed_fn


# Fixture evidence used when Neo4j is unavailable (simulates Extraction output).
FIXTURE_KG_ROWS = [
    {
        "elementName": "trust",
        "curationID": "13",
        "definition": "Willingness to be vulnerable based on positive expectations of another.",
    },
    {
        "elementName": "risk",
        "curationID": "13",
        "definition": "Uncertainty about outcomes that matter.",
    },
]

FIXTURE_CHUNKS = [
    {
        "text": (
            "We conducted a qualitative field study of trust in virtual teams. "
            "Participants described trust as willingness to share sensitive information "
            "when institutional safeguards were perceived as weak."
        ),
        "doi": "10.2307/qualitative-trust-001",
        "score": 0.88,
        "chunk_uuid": "fixture-chunk-1",
    },
    {
        "text": (
            "A case study of e-commerce platforms showed that antecedents of trust "
            "included perceived security and prior experience, while consequents "
            "included purchase intention and loyalty."
        ),
        "doi": "10.2307/qualitative-trust-002",
        "score": 0.82,
        "chunk_uuid": "fixture-chunk-2",
    },
]


@dataclass
class AgentHarness:
    client: Any
    model: str
    elicitation: ElicitationAgent
    integration: IntegrationAgent
    reflection: ReflectionAgent
    vicarious: VicariousLearningAgent
    orchestrator: KGMultiAgentOrchestrator
    embed_fn: Callable[[List[str]], List[List[float]]]
    driver: Any
    neo4j_ok: bool
    neo4j_detail: str

    def close(self) -> None:
        if self.neo4j_ok and hasattr(self.driver, "close"):
            self.driver.close()


def build_harness(*, use_neo4j: bool = True) -> AgentHarness:
    load_env()
    client = build_azure_client()
    model = os.environ.get("AZURE_MODEL_NAME", "gpt-5.2")
    embed_fn = build_embed_fn()

    neo4j_ok, neo4j_detail = probe_neo4j() if use_neo4j else (False, "disabled by test harness")
    if neo4j_ok:
        uri = os.environ["NEO4J_URI"].strip()
        user = os.environ["NEO4J_USERNAME"].strip()
        password = os.environ["NEO4J_PASSWORD"].strip()
        driver = GraphDatabase.driver(uri, auth=(user, password))
    else:
        driver = MagicMock(name="neo4j_driver_mock")

    extraction = ExtractionAgent(embed_fn)
    orchestrator = KGMultiAgentOrchestrator(
        driver,
        client,
        embed_fn,
        model=model,
        max_cycles=2,
        use_kg=True,
        log_progress=False,
    )
    return AgentHarness(
        client=client,
        model=model,
        elicitation=ElicitationAgent(client, model=model),
        integration=IntegrationAgent(client, model=model),
        reflection=ReflectionAgent(client, model=model),
        vicarious=VicariousLearningAgent(client, extraction, model=model),
        orchestrator=orchestrator,
        embed_fn=embed_fn,
        driver=driver,
        neo4j_ok=neo4j_ok,
        neo4j_detail=neo4j_detail,
    )
