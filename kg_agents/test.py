"""
Quick start for running this project with Azure OpenAI:

1) Install dependencies:
   pip install python-dotenv neo4j openai sentence-transformers scikit-learn pandas mlxtend langgraph pydantic

2) Ensure `kg_agents/.env` has your Azure credentials:
   AZURE_ENDPOINT=https://...
   AZURE_API_KEY=...
   AZURE_API_VERSION=2024-12-01-preview
   AZURE_MODEL_NAME=gpt-5.2
   AZURE_MODEL_DEPLOYMENT=gpt-5.2

3) Run from the project root:
   python -m kg_agents.test "Your question"
"""

from __future__ import annotations

import sys

from kg_agents.runtime import create_runtime


def main() -> None:
    question = ' '.join(sys.argv[1:]).strip() or 'What is the meaning of trust in this literature?'
    runtime = create_runtime()
    try:
        result = runtime.ask(question)
        print(result.answer)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
