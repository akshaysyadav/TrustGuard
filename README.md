# TrustGuard

AI-assisted trust & safety toolkit with three modules: product risk scoring,
fake review detection, and phishing URL detection.

## Architecture

React frontend → Node/Express backend → two independent FastAPI ML services:

- `ml-service/` — fake review classifier (scikit-learn) + phishing URL
  classifier (XGBoost)
- `product_scam_ml/ml-service/` — product risk scoring (Isolation Forest)

## Setup

Each service needs its own install step. Run these once.

```bash
# 1. ml-service (fake reviews + phishing)
cd ml-service
pip install -r requirements.txt

# 2. product_scam_ml (product risk)
cd ../product_scam_ml
pip install -r requirements.txt

# 3. backend
cd ../backend
npm install
cp .env.example .env   # adjust ports if needed

# 4. frontend
cd ../frontend
npm install
```

## Running (4 terminals, in this order)

```bash
# Terminal 1
cd ml-service
uvicorn src.main:app --reload --port 8000

# Terminal 2
cd product_scam_ml/ml-service
uvicorn main:app --reload --port 8001

# Terminal 3
cd backend
npm run dev

# Terminal 4
cd frontend
npm run dev
```

Then open the frontend's local URL (Vite will print it, typically
`http://localhost:5173`).

## Notes

- `product_scam_ml/data/raw/amazon_metadata_lookup.json` is required for the
  Amazon-URL lookup flow — don't remove it even though the surrounding
  `data/raw/` folder is normally gitignored.
- First run of the phishing classifier may attempt to fetch a public suffix
  list over the network; it falls back to a bundled snapshot if that fails,
  so it still works offline.
