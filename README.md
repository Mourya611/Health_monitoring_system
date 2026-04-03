# Health Monitoring System

Realtime health monitoring project with:
- React frontend dashboard
- Physiological ML risk prediction backend
- NLP PDF risk scoring + fused final risk API

## Project Structure

- `frontend/`: Vite + React dashboard UI
- `backend/`: ML pipeline, realtime predictor, NLP API, and artifacts
- `backend/artifacts/`: trained physiological model/scaler metadata
- `backend/models/`: NLP model artifacts (BioBERT and/or TF-IDF)
- `backend/data/`: runtime patient rows (`patients.csv`)

## Training Datasets

- Medical Transcriptions (NLP training): `https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions`
- Human Vital Sign Dataset (physiological model training): `https://www.kaggle.com/datasets/nasirayub2/human-vital-sign-dataset`
- Download these datasets locally into `backend/data/` before training.

## Run Steps (Windows / PowerShell)

1. Train physiological model artifacts (one-time, or when retraining)

```powershell
cd backend
python run_pipeline.py
```

2. Start physiological realtime API

```powershell
cd backend
python realtime_predict_server.py
```

3. Start NLP + final-risk API

```powershell
cd backend
python -m uvicorn nlp_api:app --host 127.0.0.1 --port 8001
```

4. Start frontend (new terminal)

```powershell
cd frontend
npm install
npm run dev
```
<img width="1920" height="1080" alt="Screenshot (587)" src="https://github.com/user-attachments/assets/79c2af28-0554-4a75-a46e-3388db89c28b" />

## Local Host Links

- Frontend: `http://127.0.0.1:5173`
- Physiological API health: `http://127.0.0.1:8000/health`
- NLP API health: `http://127.0.0.1:8001/health`

## Main API Endpoints

- `POST http://127.0.0.1:8000/predict`
- `POST http://127.0.0.1:8001/upload-pdf`
- `POST http://127.0.0.1:8001/predict-risk`

## One-Command Service Check

```powershell
cd C:\Users\Nani\OneDrive\Desktop\ML
powershell -ExecutionPolicy Bypass -File .\check-services.ps1
```

Expected output when all services are running:
- `[OK] frontend -> 200`
- `[OK] backend -> 200`
- `[OK] nlp -> 200`

## Notes

- The NLP inference supports BioBERT as primary model.
- If `backend/models/biobert` is missing, it auto-falls back to TF-IDF artifacts when available.
- Trained artifacts, runtime logs, caches, and local backups are intentionally excluded from git for a clean portfolio repository.

## Security Notes

- Do not commit API keys, tokens, or secrets to git.
- Frontend Firebase values must be provided through environment variables in `frontend/.env`.
- Use `frontend/.env.example` as the template and keep your real `.env` private.

## Realtime Data Flow

1. Frontend streams patient vitals (and optionally Firebase data).
2. Frontend sends vitals + profile to `/predict`.
3. Backend computes derived features and predicts physiological risk.
4. Backend appends prediction rows to `backend/data/patients.csv`.
5. User uploads medical PDF to NLP endpoints.
6. System combines physiological score + NLP score into final risk category.
