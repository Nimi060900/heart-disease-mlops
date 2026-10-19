# Heart Disease Risk Prediction - End-to-End MLOps

A classifier that predicts the presence of heart disease from 13 patient measurements
(UCI Heart Disease, Cleveland subset), served as a containerised, monitored API on Kubernetes.

Repository: https://github.com/Nimi060900/heart-disease-mlops

## Architecture

```
UCI data -> src/data.py -> clean CSV -> src/train.py (3 models, GridSearchCV, MLflow)
                                              |
                                      models/model.joblib (preprocessing + model)
                                              |
        GitHub Actions: flake8 -> pytest -> train      FastAPI (api/app.py) -> Docker image
                                                                 |
                                              Kubernetes (2 replicas, LoadBalancer :8080)
                                                                 |
                                        /metrics -> Prometheus -> Grafana dashboard
```

## Project layout

| Path | Purpose |
|---|---|
| `src/data.py` | Download, clean and save the dataset |
| `src/features.py` | Preprocessing `ColumnTransformer` and full pipeline |
| `src/train.py` | Tuning, cross-validation, MLflow tracking, model selection |
| `src/predict.py` | Inference helper used by the API |
| `api/app.py` | FastAPI service: `/predict`, `/health`, `/metrics` |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `tests/` | Pytest unit tests (data, model, API) |
| `k8s/` | Deployment and Service manifests |
| `monitoring/` | Prometheus + Grafana (docker-compose, provisioned dashboard) |
| `.github/workflows/ci.yml` | CI: lint, test, train, upload artifacts |
| `screenshots/` | Evidence for the report |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

### 1. Get and clean the data
```bash
python src/data.py                # downloads from UCI -> data/raw, writes data/processed/heart_clean.csv
```
The raw file is also committed, so `python src/data.py --no-download` works offline.

### 2. EDA
Install `requirements-notebooks.txt`, then open `notebooks/01_eda.ipynb`.

### 3. Train and track
```bash
python -m src.train
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```
Trains Logistic Regression, Random Forest and Gradient Boosting (5-fold stratified
`GridSearchCV`, scored on ROC-AUC), logs parameters, CV and hold-out metrics, confusion
matrix and ROC plots to MLflow, and saves the selected pipeline to `models/model.joblib`.

### 4. Test and lint
```bash
flake8 src tests api
pytest -v
```

### 5. Run the API in Docker
```bash
docker build -t heart-api:1.0 .
docker run -d --name heart-api -p 8000:8000 heart-api:1.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Response: `{"prediction":0,"confidence":0.8345,"probability_disease":0.1655}`.
Interactive docs: http://localhost:8000/docs

### 6. Deploy to Kubernetes (Docker Desktop)
Enable Kubernetes in Docker Desktop, build the image as above, then:
```bash
kubectl apply -f k8s/
kubectl get pods,svc
curl http://localhost:8080/health
```

### 7. Monitoring
```bash
cd monitoring
docker compose up -d
```
Prometheus: http://localhost:9090 (targets page shows the API as `UP`).
Grafana: http://localhost:3000 (`admin` / `admin`), dashboard "Heart Disease API".
Request logs: `kubectl logs deployment/heart-api` or `docker logs heart-api`.

## Input features

| Feature | Meaning |
|---|---|
| age, sex | Age in years; 1 = male |
| cp | Chest pain type (1-4) |
| trestbps, chol | Resting blood pressure, serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak, slope | ST depression and slope of the peak exercise ST segment |
| ca | Major vessels coloured by fluoroscopy (0-4) |
| thal | Thallium scan result (3, 6, 7) |

## Data credit

Hungarian Institute of Cardiology (Andras Janosi, M.D.), University Hospital Zurich
(William Steinbrunn, M.D.), University Hospital Basel (Matthias Pfisterer, M.D.), and
V.A. Medical Center Long Beach and Cleveland Clinic Foundation (Robert Detrano, M.D., Ph.D.).
Donor: David W. Aha. Source: UCI Machine Learning Repository.
