# KDE Agent Testing

## Quick start

From `code folder/`:

```bash
# 1) Mock unit tests (no Azure / Neo4j) — CI & regression
python3 -m pytest tests/test_kde_agents.py -v

# 2) Live quality tests (Azure required; Neo4j optional)
python3 -m pytest tests/test_kde_live_quality.py -v -s

# 3) Generate / refresh test report (auto-picks live or mock)
python3 tests/run_kde_quality_suite.py
python3 tests/run_kde_quality_suite.py --no-llm   # mock only
```

Report output: [`../docs/KDE_AGENT_TEST_REPORT.md`](../docs/KDE_AGENT_TEST_REPORT.md)

## What is tested

| Agent | Isolated? | Neo4j needed? | Quality rubric |
|-------|-----------|---------------|----------------|
| ElicitationAgent | Yes | No | refined_question, concepts, discovery_type, clarifying Qs |
| IntegrationAgent | Yes | No (fixture KG rows) | concepts, synthesis, confidence, propositions/gaps |
| ReflectionAgent | Yes | No | rationale, allowed analyses, uncertainties |
| VicariousLearningAgent | Yes | Optional | readings, excerpts, concept alignment |
| KnowledgePackage | Yes | Optional | executive synthesis, provenance, gaps, readings |

## Test cases (TC01–TC04)

1. **TC01_vague** — `trust`
2. **TC02_definition** — definitions of trust
3. **TC03_relationship** — antecedents/consequents
4. **TC04_research_gap** — theoretical gaps in virtual teams

## Grade scale

- **Good** ≥85% — structurally complete, aligned with intent
- **Acceptable** 65–84% — demo-ready with minor gaps
- **Weak** 45–64% — partial; review prompts/evidence
- **Poor** <45% — likely failure or empty synthesis
