from fastapi import FastAPI
import pickle
import pandas as pd
from sklearn.datasets import load_digits

app = FastAPI()
model = pickle.load(open("models/random_forest_model/random_forest_model_2/model.pkl", "rb"))

from fastapi import Query

@app.get("/predict")
def predict():
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=digits.feature_names)
    df['target'] = digits.target

    random_line = df.sample(n=1)

    x = random_line.drop(columns=["target"]).to_dict(orient="records")[0]
    y = model.predict(random_line.drop(columns=["target"]).values)

    return {"input": x, "prediction": float(y[0]), "actual": int(random_line["target"].iloc[0])}
