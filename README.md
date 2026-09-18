# Privacy Guardrail

A privacy-aware text processing service for detecting sensitive information, scoring risk, and applying privacy-preserving transformations.

## Structure

- `backend/`: API, NLP, risk scoring, privacy modules, database layer, and tests
- `frontend/`: client application
- `data/`: sample and synthetic datasets
- `models/`: persisted model artifacts
- `docs/`: architecture, API contract, and research notes

## Development

Create a virtual environment, install dependencies from `requirements.txt`, and run the API with:

```bash
uvicorn backend.main:app --reload
```
