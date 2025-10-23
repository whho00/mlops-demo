from fastapi import FastAPI
import pickle, numpy as np


app = FastAPI()
model = pickle.load(open('model/model.pkl', 'rb'))


@app.post("/predict")
def predict(data: list[float]):
    prediction = model.predict(np.array(data).reshape(1, -1))
    return {"prediction": prediction.tolist()}
