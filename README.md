# MLOps Project: MLflow with DagsHub and GitHub Actions

This project demonstrates a complete MLOps workflow, from model experimentation and tracking with MLflow and DagsHub to continuous integration and deployment with GitHub Actions and Docker.

The goal is to train a machine learning model, track its performance, and automatically build and deploy a Docker container serving the best model via a REST API.

## Tech Stack

*   **Python**: Core programming language.
*   **Scikit-learn**: For model training.
*   **MLflow**: For experiment tracking and model logging.
*   **DVC**: For data versioning.
*   **DagsHub**: Central hub for hosting the Git repository, DVC data, and MLflow experiment server.
*   **FastAPI**: For serving the model as a REST API.
*   **Docker**: For containerizing the application.
*   **GitHub Actions**: For CI/CD automation.

## Project Structure

```
.
├── .dvc/              # DVC metadata
├── .github/workflows/ # GitHub Actions workflows (to be created)
├── data/
│   └── raw_dataset.csv.dvc # DVC pointer to the dataset
├── src/
│   ├── app.py         # FastAPI application to serve the model
│   └── train.py       # Script for model training and logging
├── Dockerfile           # Defines the Docker image for the API
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Workflow

### 1. Experimentation and Training

The `src/train.py` script handles model training.

*   It uses the **Digits dataset** from `scikit-learn`.
*   It connects to the MLflow tracking server hosted on DagsHub to log experiments.
*   For each run, it logs:
    *   **Hyperparameters** (e.g., `n_estimators`, `max_depth`).
    *   **Metrics** (e.g., `accuracy`).
    *   **Model Artifacts** (the trained `RandomForestClassifier` model).

### 2. Data Versioning

DVC is used to version control the dataset. The `data/raw_dataset.csv.dvc` file tracks the actual data file, which is stored in DagsHub's DVC remote storage.

**Note**: The current `src/train.py` uses a dataset from `scikit-learn` directly. To use the DVC-tracked data, the script would need to be modified to read `data/raw_dataset.csv`.

### 3. CI/CD Pipeline with GitHub Actions

A GitHub Actions workflow (to be created in `.github/workflows/`) automates the deployment process. On every push to the `main` branch, the pipeline will:

1.  **Set up the environment**: Check out the code and install dependencies.
2.  **(Optional but Recommended) Download the Best Model**: A production-grade workflow would query the MLflow server on DagsHub to find the best-performing model run and download its artifacts. The current `Dockerfile` uses a static model committed to the repository.
3.  **Build Docker Image**: Build a Docker image based on the `Dockerfile`. This image contains the FastAPI application and the model.
4.  **Push to Docker Hub**: Log in to Docker Hub and push the newly built image.

### 4. API Endpoint

The `src/app.py` file defines a simple API with one endpoint:

*   `GET /predict`: This endpoint loads the `digits` dataset, selects a random row, and uses the packaged model to make a prediction. It returns a JSON object containing the input features, the model's prediction, and the actual target value for comparison.

## How to Run

### Prerequisites

*   Git
*   Python 3.10+
*   Docker
*   DVC (`pip install dvc`)

### 1. Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-github-repo-url>
    cd <repo-name>
    ```

2.  **Configure DagsHub:**
    Set up your DagsHub credentials for MLflow and DVC.
    ```bash
    dagshub login
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull DVC Data:**
    ```bash
    dvc pull
    ```

### 2. Train Models

Run the training script to log new experiments to DagsHub.

```bash
python src/train.py
```

You can view the results in the "Experiments" tab of your DagsHub repository.

### 3. Run the API Locally

To test the FastAPI application before containerizing it:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 4. Build and Run with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t your-dockerhub-username/your-repo-name .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8000:8000 your-dockerhub-username/your-repo-name
    ```

### 5. Test the Deployed API

Once the container is running (either locally or from a pulled Docker Hub image), you can test the endpoint:

```bash
curl http://localhost:8000/predict
```

You should receive a JSON response like this:

```json
{
  "input": { "pixel_0_0": 0.0, ... },
  "prediction": 5.0,
  "actual": 5
}
```
