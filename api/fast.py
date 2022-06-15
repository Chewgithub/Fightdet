from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
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
    return {"greeting": "Violence Detection System v1.0"}

@app.post("/uploadfile/")
async def upload_file(uploaded_file: UploadFile):
    start = time.perf_counter()
    filename = uploaded_file.filename

    # copy uploaded_file to a locally-saved video
    # predict on the local video
    with open(f'to_predict.mp4','wb') as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    result=make_prediction('to_predict.mp4')

    # # transfer uploaded_file to temp file for prediction
    # # store only the prediction result
    # suffix = Path(filename).suffix
    # with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    #     shutil.copyfileobj(uploaded_file.file, tmp)
    #     result=make_prediction(tmp.name)

    # close connection to uploaded_file
    uploaded_file.file.close()
    end = time.perf_counter()
    return {"filename": filename, 'result':result, 'time':round(end-start,2)}
