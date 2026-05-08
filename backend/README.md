# Backend

## Core Services

- `run_pipeline.py` : physiological ML training + artifacts
- `realtime_predict_server.py` : realtime physiological risk API (`:8000`)
- `train_nlp_models.py` : NLP model training (TF-IDF + BioBERT)
- `nlp_api.py` : PDF NLP and final risk fusion API (`:8001`)

## Data

- Physiological dataset: `data/human_vital_signs_dataset_2024.csv`
- NLP dataset: `data/mtsamples.csv` (fallback resolver also checks `data/empty_samples`)

## NLP Training

```powershell
python train_nlp_models.py
```

Saved outputs:
- `models/tfidf/*`
- `models/biobert/*`
- `models/model_comparison.json`

## Run APIs

Physiological API:
```powershell
python realtime_predict_server.py
```

NLP + final fusion API:
```powershell
uvicorn nlp_api:app --host 127.0.0.1 --port 8001
```

## Optional Gemini PDF Analysis

If `GEMINI_API_KEY` is set, uploaded PDFs are sent directly to Gemini for structured medical summarization before NLP risk scoring. This is useful for scanned/image-heavy reports and can avoid local OCR delays.

Optional environment variables:

```powershell
$env:GEMINI_API_KEY="your-key"
$env:GEMINI_MODEL="gemini-2.5-flash"
```

## NLP Endpoints

- `POST /upload-pdf`
- `POST /predict-risk`
- `GET /health`
