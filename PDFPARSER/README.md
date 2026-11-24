# Resume Parser (Colab prototype)

This project follows `../RESUME_PARSER_COLAB_IMPLEMENTATION_PLAN (1).md`. The
current implementation covers **Phase P0 through P2 (RP-001 through RP-009)**:
project foundation, configuration, lifecycle, strict resume schemas, versioned
prompts, model-independent generation contracts, deterministic mock generation,
strict output processing, validation, bounded retry, bounded PDF/DOCX text
extraction, HTTP parsing endpoints, and CPU-only tests.
Fresh-Colab verification remains pending.

Mock inference is the default. Importing `app.main` does not import PyTorch,
initialize CUDA, download a checkpoint, or read private resume data. Real model
loading belongs to a later reviewed phase. PDF extraction uses `pypdf` and DOCX
extraction uses `python-docx`; OCR and complex layout reconstruction are not
part of V1.

The P1 parsing core can be exercised directly with the deterministic mock:

```python
from app.services.llm_service import MockLLMService
from app.services.parsing_service import ParsingService

result = ParsingService(MockLLMService()).parse("Kỹ năng: Python")
print(result.model_dump(mode="json"))
```

## Local setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
uvicorn app.main:app --reload
```

Then call:

```text
GET http://127.0.0.1:8000/api/v1/health
GET http://127.0.0.1:8000/api/v1/ready
```

Parse plain text or upload a PDF/DOCX document:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/resumes/parse-text \
  -H "Content-Type: application/json" \
  -d '{"text":"Kỹ năng: Python"}'

curl -X POST http://127.0.0.1:8000/api/v1/resumes/parse \
  -F "file=@resume.pdf"
```

Liveness only confirms that the process is running. Readiness confirms that the
configured inference implementation is usable. Mock mode is ready immediately;
transformers mode stays unready until the Phase P3 Model Manager is implemented.

## Google Colab setup

Use a fresh runtime and replace the repository URL below.

```python
!git clone <YOUR_REPOSITORY_URL> /content/ATS-CV
%cd /content/ATS-CV/PDFPARSER
%pip install -r requirements-dev.txt
```

Run the CPU/mock verification without any model download:

```python
!python -m pytest

from fastapi.testclient import TestClient
from app.main import create_app

with TestClient(create_app()) as client:
    print(client.get("/api/v1/health").json())
    print(client.get("/api/v1/ready").json())
```

The reproducible Phase P2 verification entry point is
`notebooks/02_test_resume_parser.ipynb`. It generates only synthetic document
fixtures and never uploads private resumes.

Phase P3 begins with `notebooks/01_setup_colab.ipynb`. It installs the ML
dependencies, configures an ephemeral Hugging Face cache, checks GPU/RAM/disk
and package versions, performs a small bitsandbytes NF4 operation, and saves an
explicit `supported_for_load_test` or `blocked` report. It does not download a
checkpoint or silently switch to a smaller model.

Do not install `requirements-ml.txt` for P0. When Phase P3 is reviewed and
implemented, the Colab runtime must perform a GPU/resource preflight before real
model loading. No public tunnel or always-on deployment is part of V1.

## Configuration

Copy `.env.example` to `.env` when environment overrides are needed. Important
settings include inference/model selection, model revision and quantization,
input/output caps, retry cap, file/page limits, artifact path and log level.
Invalid combinations fail during application creation.

API errors follow this stable envelope:

```json
{
  "request_id": "req_example",
  "status": "error",
  "error": {"code": "MODEL_NOT_READY", "message": "Model is not ready"}
}
```

Logs contain request metadata and error codes only. Resume text, prompts, model
output, secrets and stack traces are never returned in API responses.
