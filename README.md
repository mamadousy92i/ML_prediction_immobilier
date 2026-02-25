
# Immo Predictor API

API de Machine Learning pour l'immobilier, construite avec **FastAPI** + **Gradio**.

| Fonctionnalite | Description |
|---|---|
| **Regression** | Prediction du prix de vente (`SalePrice`) |
| **Classification** | Classification du type de batiment (`BldgType`) |

**Auteur** : [mamadousy92i](https://github.com/mamadousy92i)

---

## Lancer en local

```bash
# Creer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dependances
pip install -r requirements.txt

# Demarrer le serveur
python -m uvicorn app.main:app --reload --port 8000
```

- Interface Utilisateur : [http://localhost:8000/ui](http://localhost:8000/ui)
- Documentation Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Endpoints

### `GET /` — Message de bienvenue

```bash
curl http://localhost:8000/
```

### `GET /health` — Statut de l'API

```bash
curl http://localhost:8000/health
```

### `GET /models/info` — Informations sur les modeles

```bash
curl http://localhost:8000/models/info
```

### `POST /regression/predict` — Predire le prix

**Parametre query** : `model` = `random_forest` (defaut) | `decision_tree`

```bash
curl -X POST "http://localhost:8000/regression/predict?model=random_forest" \
  -H "Content-Type: application/json" \
  -d '{
    "GrLivArea": 1500,
    "TotalBsmtSF": 800,
    "LotArea": 9000,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "TotRmsAbvGrd": 7,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2000,
    "YearRemodAdd": 2005,
    "Neighborhood": "NAmes",
    "GarageCars": 2,
    "GarageArea": 500,
    "PoolArea": 0,
    "Fireplaces": 1
  }'
```

**Reponse** :
```json
{
  "model_used": "random_forest",
  "predicted_price": 185000.0,
  "currency": "USD"
}
```

### `POST /classification/predict` — Classifier le type de bien

**Parametre query** : `model` = `random_forest` (defaut) | `svm`

```bash
curl -X POST "http://localhost:8000/classification/predict?model=svm" \
  -H "Content-Type: application/json" \
  -d '{
    "GrLivArea": 1500,
    "TotRmsAbvGrd": 7,
    "OverallQual": 7,
    "YearBuilt": 2000,
    "GarageCars": 2,
    "Neighborhood": "NAmes",
    "HouseStyle": "1Story"
  }'
```

**Reponse** :
```json
{
  "model_used": "svm",
  "predicted_type": "1Fam",
  "predicted_type_encoded": 0
}
```

---

## Docker (HuggingFace Spaces)

```bash
docker build -t immo-predictor .
docker run -p 7860:7860 immo-predictor
```

---

## Demo HuggingFace

[https://lucifer92i-immo-predictor.hf.space/ui/](https://lucifer92i-immo-predictor.hf.space/ui/)

---

## GitHub

[https://github.com/mamadousy92i](https://github.com/mamadousy92i)
