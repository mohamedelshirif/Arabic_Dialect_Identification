from __future__ import annotations
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nlp.nlp import Trainer_Dl,Trainer_Ml

app = FastAPI()
trainer_ml = Trainer_Ml()
trainer_dl = Trainer_Dl()

class TrainingData(BaseModel):
    texts: List[str]
    labels: List[Union[str, int]]

class TestingData(BaseModel):
    texts: List[str]

class QueryText(BaseModel):
    text: str

class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes: List[str]
    evaluation: Dict

class PredictionObject(BaseModel):
    text: str
    predictions: Dict

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]


@app.get("/status_ml", summary="Get current status of the system")
def get_status():
    status = trainer_ml.get_status()
    return StatusObject(**status)

@app.post("/train_ml", summary="Train a new model")
def train(training_datafram:TrainingData):
    try:
        trainer_ml.train(training_datafram)
        status = trainer_ml.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict_ml", summary="Predict list of inputs (one or more)")
def predict(query_list: QueryText):
    try:
        prediction = trainer_ml.predict(query_list)
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    
@app.get("/status_dl", summary="Get current status of the system")
def get_status():
    status = trainer_dl.get_status()
    return StatusObject(**status)

@app.post("/train_dl", summary="Train a new model")
def train(dataframe:TrainingData):
    try:
        trainer_dl.train(dataframe)
        status = trainer_dl.get_status()
        return StatusObject(**status)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict_dl", summary="Predict list of inputs (one or more)")
def predict(query_list: QueryText):
    try:
        prediction = trainer_dl.predict(query_list)
        return PredictionObject(**prediction)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))



@app.get("/")
def home():
    return({"message": "System is up"})
