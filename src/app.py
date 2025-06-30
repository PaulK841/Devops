# src/app.py

import os
import pickle
import requests
import mlflow
import dagshub
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Modèle de données pour la requête d'entrée ---
class PredictionRequest(BaseModel):
    data: list[float]

# --- Configuration ---
REPO_OWNER = 'paulker194'
REPO_NAME = 'mlops'
EXPERIMENT_NAME = "RandomForestExperiment_2"
MODEL = None

# --- Gestionnaire de cycle de vie pour le démarrage de l'API ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print("--- Démarrage de l'API : téléchargement du modèle depuis DagsHub... ---")
    
    # ÉTAPE 1: Lire le token d'accès depuis les variables d'environnement
    # C'est la manière sécurisée de gérer les secrets.
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        print("!!! ERREUR FATALE : La variable d'environnement DAGSHUB_TOKEN est manquante. !!!")
        # L'API ne peut pas fonctionner sans modèle.
        yield # Permet à l'API de démarrer, mais le modèle sera None
        return

    try:
        # ÉTAPE 2: Se connecter à MLflow pour trouver le meilleur modèle
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["metrics.accuracy DESC"], max_results=1)
        if runs.empty:
            raise RuntimeError(f"Aucun run trouvé pour l'expérience '{EXPERIMENT_NAME}'.")

        best_run_id = runs.iloc[0].run_id
        experiment_id = runs.iloc[0].experiment_id # Important pour construire le chemin
        print(f"Meilleur run trouvé : {best_run_id}")
        
        # ÉTAPE 3: Construire l'URL de téléchargement direct de l'artefact
        # C'est une URL d'API directe pour le contenu brut du fichier
        model_url = f"https://dagshub.com/api/v1/repos/{REPO_OWNER}/{REPO_NAME}/dvc/files/DVC/.mlflow/mlruns/{experiment_id}/{best_run_id}/artifacts/model/model.pkl"
        
        print(f"Téléchargement du modèle depuis l'URL : {model_url}")
        
        # ÉTAPE 4: Faire la requête HTTP avec authentification
        headers = {"Authorization": f"Bearer {dagshub_token}"}
        response = requests.get(model_url, headers=headers)
        response.raise_for_status()  # Lève une exception si l'URL est mauvaise ou l'accès refusé (404, 403, etc.)
        
        # ÉTAPE 5: Charger le modèle depuis le contenu de la réponse
        MODEL = pickle.loads(response.content)
        print("--- Modèle chargé avec succès. API prête. ---")

    except Exception as e:
        print(f"!!! ERREUR FATALE lors du chargement du modèle : {e} !!!")
        MODEL = None
    
    yield
    print("--- Arrêt de l'API. ---")

# --- Initialisation de l'application FastAPI ---
app = FastAPI(title="Digits Classifier API", version="1.0", lifespan=lifespan)

# --- Endpoints (inchangés) ---
@app.get("/")
def read_root(): return {"status": "API en ligne", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None: raise HTTPException(status_code=503, detail="Modèle non disponible.")
    if len(request.data) != 64: raise HTTPException(status_code=400, detail="Les données d'entrée doivent contenir 64 éléments.")
    try:
        input_data = np.array(request.data).reshape(1, -1)
        prediction_result = MODEL.predict(input_data)
        return {"prediction": int(prediction_result[0])}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")