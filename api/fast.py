from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import tempfile
import shutil
from Fightdet.predict import make_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

{
  "greeting": "Hello world"
}

@app.get("/predict")
def predict(aa, bb):
    # compute `wait_prediction` from `day_of_week` and `time`
    return aa+bb

@app.post("/uploadfile/")
async def create_upload_file(uploaded_video: UploadFile):
    with open(f'to_predict.mp4','wb') as buffer:
        shutil.copyfileobj(uploaded_video.file, buffer)
    result=make_prediction('to_predict.mp4')


    return {"filename": uploaded_video.filename, 'Result':result}
